"""Interleaved Clifford Randomised Benchmarking Average Gate Error Metric.

This module provides the Interleaved Clifford randomised benchmarking average gate
error implementation for the QCMet framework. This metric provides an
estimate for the average gate error of a target Clifford gate in a gate set.
Here the benchmarking procedure follows M3.4 from arxiv:2502.06717.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from qiskit import QuantumCircuit

    from qcmet.core import FileManager

from qcmet.benchmarks import BaseBenchmark
from qcmet.benchmarks.gate_execution_quality_metrics import CliffordRB
from qcmet.benchmarks.gate_execution_quality_metrics.rb_fitting import (
    RBFitConfig,
    average_survival_by_m,
    fit_curve,
    get_joint_log_initial_guess,
    get_joint_rb_fit_model,
)


class InterleavedRB(BaseBenchmark):
    """Implements Interleaved Clifford Randomised Benchmarking Average Gate Error Metric.

    This class generates both standard Clifford RB circuits and interleaved Clifford
    RB circuits, measures the output, and computes the average gate error of a
    specific target Clifford gate.

    The final analysis uses a linked joint fit:

        RB:  p_survival(m) = a0 * alpha**m   + b0
        IRB: p_survival(m) = a0 * alpha_g**m + b0

    where a0 and b0 are shared between the RB and IRB curves.
    """

    def __init__(
        self,
        m_list: List[int],
        target_clifford: QuantumCircuit,
        circs_per_m: int = 5,
        qubits: int | List[int] = 1,
        seed: int | None = None,
        save_path: str | Path | FileManager | None = None,
        fixed_a0: bool = False,
        fixed_b0: bool = False,
        fit_method: str = "two_step",
        errorbar_method: str = "bootstrap",
        bootstrap_samples: int = 1000,
        bootstrap_seed: int | None = None,
        bootstrap_confidence_level: float = 0.68,
    ):
        """Initialize the Interleaved Clifford randomised benchmark.

        Args:
            m_list (list): Sequence lengths to run the benchmark on.
            target_clifford (QuantumCircuit): Target Clifford gate to interleave.
            circs_per_m (int): Number of random circuits per sequence length.
            qubits (int | List[int]): Number of qubits or list of qubit indices.
            seed (int | None): Seed for reproducible random Clifford generation.
            save_path (str | Path | FileManager | None): Directory path to save results.
            fixed_a0 (bool): If True, constrain a0 = 1 - b0.
            fixed_b0 (bool): If True, fix b0 = 1 / 2**num_qubits in the final fit.
            fit_method (str): Either "two_step" or "curve_fit".
                - "two_step": use log-linear initialisation, then joint curve_fit.
                - "curve_fit": use joint curve_fit directly.
               
                Note: A pure "log" fit is not supported for linked InterleavedRB,
                because the linked fit is defined as a joint nonlinear fit.

            errorbar_method (str): One of "fit", "bootstrap", or "none".
            bootstrap_samples (int): Number of bootstrap resamples.
            bootstrap_seed (int | None): Seed for bootstrap resampling.
            bootstrap_confidence_level (float): Central bootstrap confidence level.
                For example, 0.68 gives a central 68% interval, and 0.95 gives
                a central 95% interval.

        """
        super().__init__("InterleavedRB", qubits=qubits, save_path=save_path)

        if target_clifford is None:
            raise ValueError("target_clifford needs to be specified.")

        if fit_method not in {"two_step", "curve_fit"}:
            raise ValueError(
                "Linked InterleavedRB supports fit_method='two_step' or "
                "fit_method='curve_fit'. Pure fit_method='log' is not supported."
            )

        if errorbar_method not in {"fit", "bootstrap", "none"}:
            raise ValueError(
                "errorbar_method must be one of 'fit', 'bootstrap', or 'none'."
            )

        if bootstrap_samples <= 1:
            raise ValueError("bootstrap_samples must be greater than 1.")

        if not 0 < bootstrap_confidence_level < 1:
            raise ValueError("bootstrap_confidence_level must be between 0 and 1.")

        self.config["m_list"] = m_list
        self.config["seed"] = seed
        self.config["circs_per_m"] = circs_per_m
        self.config["fixed_a0"] = fixed_a0
        self.config["fixed_b0"] = fixed_b0
        self.config["fit_method"] = fit_method
        self.config["errorbar_method"] = errorbar_method
        self.config["bootstrap_samples"] = bootstrap_samples
        self.config["bootstrap_seed"] = bootstrap_seed
        self.config["bootstrap_confidence_level"] = bootstrap_confidence_level
        self.config["target_clifford"] = [
            (gate, count) for gate, count in target_clifford.count_ops().items()
        ]

        # These child CliffordRB instances are used for circuit generation and
        # data handling. The final fit is performed jointly in this class.
        #
        # Note:
        # Your current CliffordRB class uses bootstrap_ci_level, not
        # bootstrap_confidence_level. If you later rename that argument in
        # CliffordRB, update these two calls accordingly.
        self.rb_experiment = CliffordRB(
            m_list=m_list,
            circs_per_m=circs_per_m,
            qubits=qubits,
            seed=seed,
            target_clifford=None,
            save_path=save_path,
            fixed_a0=fixed_a0,
            fixed_b0=fixed_b0,
            fit_method=fit_method,
            errorbar_method=errorbar_method,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            bootstrap_ci_level=bootstrap_confidence_level,
        )

        self.irb_experiment = CliffordRB(
            m_list=m_list,
            circs_per_m=circs_per_m,
            qubits=qubits,
            seed=seed,
            target_clifford=target_clifford,
            save_path=save_path,
            fixed_a0=fixed_a0,
            fixed_b0=fixed_b0,
            fit_method=fit_method,
            errorbar_method=errorbar_method,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            bootstrap_ci_level=bootstrap_confidence_level,
        )

    def _generate_circuits(self):
        """Generate circuits for standard and interleaved Clifford RB."""
        rb_circs = self.rb_experiment._generate_circuits()
        irb_circs = self.irb_experiment._generate_circuits()

        self.rb_experiment._experiment_data = rb_circs
        self.irb_experiment._experiment_data = irb_circs

        return rb_circs + irb_circs

    def _make_fit_config(self):
        """Construct an RBFitConfig from this benchmark's config dictionary."""
        return RBFitConfig(
            num_qubits=self.num_qubits,
            fixed_a0=self.config.get("fixed_a0", False),
            fixed_b0=self.config.get("fixed_b0", False),
            fit_method=self.config.get("fit_method", "two_step"),
            errorbar_method=self.config.get("errorbar_method", "fit"),
            bootstrap_samples=self.config.get("bootstrap_samples", 1000),
            bootstrap_seed=self.config.get("bootstrap_seed", None),
            bootstrap_confidence_level=self.config.get(
                "bootstrap_confidence_level",
                0.68,
            ),
        )

    def _joint_xdata(self, m_rb, m_irb):
        """Return xdata tuple used by the linked joint fit."""
        m_values = np.concatenate([m_rb, m_irb])
        is_irb = np.concatenate(
            [
                np.zeros_like(m_rb, dtype=bool),
                np.ones_like(m_irb, dtype=bool),
            ]
        )

        return m_values, is_irb

    def _fit_joint_sample(self, m_rb, p_rb, m_irb, p_irb, config):
        """Fit one joint RB/IRB sample.

        This is used both for the main fit and for bootstrap resamples.
        """
        m_values, is_irb = self._joint_xdata(m_rb, m_irb)
        p_surv = np.concatenate([p_rb, p_irb])

        popt, pcov, fit_params, log_fit_result = fit_curve(
            xdata=(m_values, is_irb),
            ydata=p_surv,
            fit_func=self._fit_func,
            param_names=self._param_names,
            bounds=self._bounds,
            p0=self._p0,
            config=config,
            log_initial_guess_func=lambda: get_joint_log_initial_guess(
                m_rb,
                p_rb,
                m_irb,
                p_irb,
                config,
                self._param_names,
                self._bounds,
                self._p0,
            ),
        )

        return popt, pcov, fit_params, log_fit_result

    def _get_joint_fit_covariance_errors(self, popt, pcov, param_names, config):
        """Compute linked IRB error bars from the joint fit covariance matrix."""
        if pcov is None:
            return {
                "method": "fit",
                "status": "failed",
                "reason": "Fit covariance matrix is None.",
            }

        pcov = np.asarray(pcov, dtype=float)

        if pcov.ndim != 2 or pcov.shape[0] != pcov.shape[1]:
            return {
                "method": "fit",
                "status": "failed",
                "reason": "Fit covariance matrix has invalid shape.",
            }

        variances = np.diag(pcov)

        if np.any(variances < 0):
            return {
                "method": "fit",
                "status": "failed",
                "reason": "Fit covariance matrix contains negative variances.",
            }

        param_stderr = np.sqrt(variances)
        param_errors = dict(zip(param_names, param_stderr))
        fit_params = dict(zip(param_names, popt))

        alpha = float(fit_params["alpha"])
        alpha_g = float(fit_params["alpha_g"])

        if alpha == 0:
            return {
                "method": "fit",
                "status": "failed",
                "reason": "Cannot propagate IRB uncertainty because alpha is zero.",
            }

        alpha_index = param_names.index("alpha")
        alpha_g_index = param_names.index("alpha_g")

        d = 2**config.num_qubits
        prefactor = (d - 1) / d

        avg_gate_err_stderr = prefactor * param_errors["alpha"]

        # Interleaved gate error:
        #
        #   e_g = (d - 1) / d * (1 - alpha_g / alpha)
        #
        # Gradient wrt alpha and alpha_g:
        #
        #   de/dalpha   = prefactor * alpha_g / alpha**2
        #   de/dalpha_g = -prefactor / alpha
        grad = np.zeros(len(param_names))
        grad[alpha_index] = prefactor * alpha_g / alpha**2
        grad[alpha_g_index] = -prefactor / alpha

        int_gate_err_var = grad @ pcov @ grad
        int_gate_err_stderr = np.sqrt(max(float(int_gate_err_var), 0.0))

        return {
            "method": "fit",
            "status": "success",
            "param_stderr": param_errors,
            "alpha_stderr": float(param_errors["alpha"]),
            "alpha_g_stderr": float(param_errors["alpha_g"]),
            "AverageGateError_stderr": float(avg_gate_err_stderr),
            "InterleavedGateError_stderr": float(int_gate_err_stderr),
        }

    def _summarise_bootstrap_values(self, values, confidence_level):
        """Return stderr and central percentile interval for bootstrap values."""
        values = np.asarray(values, dtype=float)

        lower_percentile = 100 * (1 - confidence_level) / 2
        median_percentile = 50.0
        upper_percentile = 100 * (1 + confidence_level) / 2

        ci = np.percentile(
            values,
            [lower_percentile, median_percentile, upper_percentile],
        )

        return {
            "stderr": float(np.std(values, ddof=1)),
            "ci": [float(x) for x in ci],
            "ci_lower_error": float(ci[1] - ci[0]),
            "ci_upper_error": float(ci[2] - ci[1]),
        }

    def _get_joint_bootstrap_errors(self, m_rb, m_irb, config):
        """Bootstrap uncertainty estimates for the linked joint RB/IRB fit."""
        rng = np.random.default_rng(config.bootstrap_seed)
        confidence_level = config.bootstrap_confidence_level

        rb_grouped = {
            m: group["p_survival"].to_numpy(dtype=float)
            for m, group in self.rb_experiment._experiment_data.groupby("m")
        }

        irb_grouped = {
            m: group["p_survival"].to_numpy(dtype=float)
            for m, group in self.irb_experiment._experiment_data.groupby("m")
        }

        bootstrap_popt = []
        bootstrap_alpha = []
        bootstrap_alpha_g = []
        bootstrap_avg_gate_err = []
        bootstrap_int_gate_err = []

        failed_fits = 0
        d = 2**self.num_qubits

        for _ in range(config.bootstrap_samples):
            try:
                boot_p_rb = []
                boot_p_irb = []

                for m in m_rb:
                    values = rb_grouped[m]

                    if len(values) == 0:
                        raise ValueError("Empty RB bootstrap group.")

                    resampled = rng.choice(values, size=len(values), replace=True)
                    boot_p_rb.append(np.mean(resampled))

                for m in m_irb:
                    values = irb_grouped[m]

                    if len(values) == 0:
                        raise ValueError("Empty IRB bootstrap group.")

                    resampled = rng.choice(values, size=len(values), replace=True)
                    boot_p_irb.append(np.mean(resampled))

                boot_p_rb = np.asarray(boot_p_rb, dtype=float)
                boot_p_irb = np.asarray(boot_p_irb, dtype=float)

                popt_i, _pcov_i, fit_params_i, _log_fit_result_i = (
                    self._fit_joint_sample(
                        m_rb=m_rb,
                        p_rb=boot_p_rb,
                        m_irb=m_irb,
                        p_irb=boot_p_irb,
                        config=config,
                    )
                )

                alpha_i = float(fit_params_i["alpha"])
                alpha_g_i = float(fit_params_i["alpha_g"])

                if alpha_i == 0:
                    raise ValueError("Bootstrap alpha is zero.")

                avg_gate_err_i = (1 - alpha_i) * (d - 1) / d
                int_gate_err_i = (d - 1) * (1 - alpha_g_i / alpha_i) / d

                bootstrap_popt.append(popt_i)
                bootstrap_alpha.append(alpha_i)
                bootstrap_alpha_g.append(alpha_g_i)
                bootstrap_avg_gate_err.append(avg_gate_err_i)
                bootstrap_int_gate_err.append(int_gate_err_i)

            except Exception:
                failed_fits += 1
                continue

        n_successful = len(bootstrap_alpha)

        if n_successful < 2:
            return {
                "method": "bootstrap",
                "status": "failed",
                "reason": "Fewer than two bootstrap joint fits succeeded.",
                "n_bootstrap": config.bootstrap_samples,
                "n_successful": n_successful,
                "n_failed": failed_fits,
                "confidence_level": confidence_level,
            }

        bootstrap_popt = np.asarray(bootstrap_popt, dtype=float)

        param_stderr = np.std(bootstrap_popt, axis=0, ddof=1)
        param_errors = dict(zip(self._param_names, param_stderr))

        alpha_summary = self._summarise_bootstrap_values(
            bootstrap_alpha,
            confidence_level,
        )
        alpha_g_summary = self._summarise_bootstrap_values(
            bootstrap_alpha_g,
            confidence_level,
        )
        avg_gate_err_summary = self._summarise_bootstrap_values(
            bootstrap_avg_gate_err,
            confidence_level,
        )
        int_gate_err_summary = self._summarise_bootstrap_values(
            bootstrap_int_gate_err,
            confidence_level,
        )

        lower_percentile = 100 * (1 - confidence_level) / 2
        upper_percentile = 100 * (1 + confidence_level) / 2

        return {
            "method": "bootstrap",
            "status": "success",
            "n_bootstrap": config.bootstrap_samples,
            "n_successful": n_successful,
            "n_failed": failed_fits,
            "confidence_level": confidence_level,
            "ci_percentiles": [
                float(lower_percentile),
                50.0,
                float(upper_percentile),
            ],
            "param_stderr": param_errors,
            "alpha_stderr": alpha_summary["stderr"],
            "alpha_ci": alpha_summary["ci"],
            "alpha_ci_lower_error": alpha_summary["ci_lower_error"],
            "alpha_ci_upper_error": alpha_summary["ci_upper_error"],
            "alpha_g_stderr": alpha_g_summary["stderr"],
            "alpha_g_ci": alpha_g_summary["ci"],
            "alpha_g_ci_lower_error": alpha_g_summary["ci_lower_error"],
            "alpha_g_ci_upper_error": alpha_g_summary["ci_upper_error"],
            "AverageGateError_stderr": avg_gate_err_summary["stderr"],
            "AverageGateError_ci": avg_gate_err_summary["ci"],
            "AverageGateError_ci_lower_error": avg_gate_err_summary[
                "ci_lower_error"
            ],
            "AverageGateError_ci_upper_error": avg_gate_err_summary[
                "ci_upper_error"
            ],
            "InterleavedGateError_stderr": int_gate_err_summary["stderr"],
            "InterleavedGateError_ci": int_gate_err_summary["ci"],
            "InterleavedGateError_ci_lower_error": int_gate_err_summary[
                "ci_lower_error"
            ],
            "InterleavedGateError_ci_upper_error": int_gate_err_summary[
                "ci_upper_error"
            ],

            "bootstrap_popt_samples": bootstrap_popt.tolist(),
        }

    def _get_shared_parameter_names(self):
        """Return names of parameters shared between RB and IRB."""
        fixed_a0 = self.config.get("fixed_a0", False)
        fixed_b0 = self.config.get("fixed_b0", False)

        shared = []

        if fixed_a0:
            shared.append("a0 = 1 - b0")
        else:
            shared.append("a0")

        if fixed_b0:
            shared.append("b0 = 1 / 2**num_qubits")
        else:
            shared.append("b0")

        return shared

    def _analyze(self):
        """Analyze measurement results using a linked joint RB/IRB fit."""
        config = self._make_fit_config()

        if "type" not in self._experiment_data.columns:
            raise KeyError(
                "InterleavedRB analysis requires a 'type' column distinguishing "
                "RB and IRB circuits."
            )

        if "circuit_measurements" not in self._experiment_data.columns:
            raise KeyError(
                "InterleavedRB analysis requires a 'circuit_measurements' column. "
                "Make sure the benchmark has been run before calling analyze()."
            )

        rb_data = self._experiment_data[self._experiment_data["type"] == "RB"].copy()
        irb_data = self._experiment_data[self._experiment_data["type"] == "IRB"].copy()

        if rb_data.empty:
            raise ValueError("No RB measurement data found in experiment data.")

        if irb_data.empty:
            raise ValueError("No IRB measurement data found in experiment data.")

        self.rb_experiment._runtime_params = self._runtime_params
        self.irb_experiment._runtime_params = self._runtime_params

        rb_df, self.rb_experiment._experiment_data = average_survival_by_m(
            rb_data,
            self.num_qubits,
        )

        irb_df, self.irb_experiment._experiment_data = average_survival_by_m(
            irb_data,
            self.num_qubits,
        )

        m_rb = rb_df["m"].to_numpy(dtype=float)
        p_rb = rb_df["p_survival"].to_numpy(dtype=float)

        m_irb = irb_df["m"].to_numpy(dtype=float)
        p_irb = irb_df["p_survival"].to_numpy(dtype=float)

        self.rb_m_values = m_rb
        self.rb_p_surv = p_rb
        self.irb_m_values = m_irb
        self.irb_p_surv = p_irb

        self._fit_func, self._param_names, self._bounds, self._p0 = (
            get_joint_rb_fit_model(config)
        )

        popt, pcov, fit_params, log_fit_result = self._fit_joint_sample(
            m_rb=m_rb,
            p_rb=p_rb,
            m_irb=m_irb,
            p_irb=p_irb,
            config=config,
        )

        alpha = float(fit_params["alpha"])
        alpha_g = float(fit_params["alpha_g"])

        if alpha == 0:
            raise ValueError(
                "Cannot calculate interleaved gate error because alpha is zero."
            )

        d = 2**self.num_qubits

        self.avg_gate_err = (1 - alpha) * (d - 1) / d
        self.int_gate_err = (d - 1) * (1 - alpha_g / alpha) / d

        if config.errorbar_method == "fit":
            errorbars = self._get_joint_fit_covariance_errors(
                popt=popt,
                pcov=pcov,
                param_names=self._param_names,
                config=config,
            )

        elif config.errorbar_method == "bootstrap":
            errorbars = self._get_joint_bootstrap_errors(
                m_rb=m_rb,
                m_irb=m_irb,
                config=config,
            )

        else:
            errorbars = {
                "method": "none",
                "status": "skipped",
            }

        fit_result = {
            "popt": popt,
            "pcov": pcov,
            "param_names": self._param_names,
            "params": fit_params,
            "fit_method": config.fit_method,
            "final_fit_method": "joint_curve_fit",
            "shared_parameters": self._get_shared_parameter_names(),
            "errorbar_method": config.errorbar_method,
            "errorbars": errorbars,
        }

        if log_fit_result is not None:
            fit_result["initial_fit"] = {
                "fit_method": "joint_log_linear",
                **log_fit_result,
            }

        self.fit_result = {
            "fit_result": fit_result,
        }

        self.run_id = self.file_manager.run_id if self.file_manager else None

        self.result = {
            "qubits": self.num_qubits,
            "alpha": alpha,
            "alpha_g": alpha_g,
            "alpha_stderr": errorbars.get("alpha_stderr"),
            "alpha_g_stderr": errorbars.get("alpha_g_stderr"),
            "AverageGateError": self.avg_gate_err,
            "AverageGateError_stderr": errorbars.get("AverageGateError_stderr"),
            "InterleavedGateError": self.int_gate_err,
            "InterleavedGateError_stderr": errorbars.get(
                "InterleavedGateError_stderr"
            ),
        } | self.fit_result

        return self.result

    def _plot(self, axes):
        """Plot standard and interleaved RB survival probabilities and joint fits.

        If bootstrap uncertainty estimation is enabled, shaded confidence bands are
        drawn around both the RB and IRB fitted curves using the bootstrap fitted
        parameter samples.
        """
        axes.set_xlim((0, max(self.config["m_list"])))
        axes.set_ylim((1 / 2**self.num_qubits - 0.05, 1))

        axes.plot(
            self.rb_m_values,
            self.rb_p_surv,
            linestyle="",
            marker="x",
            c="black",
            label=f"{self._runtime_params['device'].name} RB results",
        )

        axes.plot(
            self.irb_m_values,
            self.irb_p_surv,
            linestyle="",
            marker="x",
            c="green",
            label=f"{self._runtime_params['device'].name} IRB results",
        )

        fit_xxs = np.linspace(0, max(self.config["m_list"]) + 1, 1000)

        rb_is_irb = np.zeros_like(fit_xxs, dtype=bool)
        irb_is_irb = np.ones_like(fit_xxs, dtype=bool)

        popt = self.fit_result["fit_result"]["popt"]

        rb_fit = self._fit_func((fit_xxs, rb_is_irb), *popt)
        irb_fit = self._fit_func((fit_xxs, irb_is_irb), *popt)

        axes.plot(
            fit_xxs,
            rb_fit,
            linestyle="--",
            marker="",
            c="black",
            label="Joint RB fit",
        )

        axes.plot(
            fit_xxs,
            irb_fit,
            linestyle="--",
            marker="",
            c="green",
            label="Joint IRB fit",
        )

        # --- Bootstrap confidence bands ---
        errorbars = self.fit_result["fit_result"].get("errorbars", {})

        if (
            self.config.get("errorbar_method") == "bootstrap"
            and errorbars.get("status") == "success"
            and "bootstrap_popt_samples" in errorbars
        ):
            popt_samples = np.asarray(
                errorbars["bootstrap_popt_samples"],
                dtype=float,
            )

            confidence_level = self.config.get("bootstrap_confidence_level", 0.68)

            lower_percentile = 100 * (1 - confidence_level) / 2
            upper_percentile = 100 * (1 + confidence_level) / 2

            rb_bootstrap_curves = []
            irb_bootstrap_curves = []

            for popt_i in popt_samples:
                rb_bootstrap_curves.append(
                    self._fit_func((fit_xxs, rb_is_irb), *popt_i)
                )
                irb_bootstrap_curves.append(
                    self._fit_func((fit_xxs, irb_is_irb), *popt_i)
                )

            rb_bootstrap_curves = np.asarray(rb_bootstrap_curves, dtype=float)
            irb_bootstrap_curves = np.asarray(irb_bootstrap_curves, dtype=float)

            rb_lower = np.percentile(
                rb_bootstrap_curves,
                lower_percentile,
                axis=0,
            )
            rb_upper = np.percentile(
                rb_bootstrap_curves,
                upper_percentile,
                axis=0,
            )

            irb_lower = np.percentile(
                irb_bootstrap_curves,
                lower_percentile,
                axis=0,
            )
            irb_upper = np.percentile(
                irb_bootstrap_curves,
                upper_percentile,
                axis=0,
            )

            axes.fill_between(
                fit_xxs,
                rb_lower,
                rb_upper,
                color="black",
                alpha=0.2,
                label="RB bootstrap CI",
            )

            axes.fill_between(
                fit_xxs,
                irb_lower,
                irb_upper,
                color="green",
                alpha=0.2,
                label="IRB bootstrap CI",
            )

        axes.set_xlabel(r"$m$")
        axes.set_ylabel(r"$p_0$")

        return axes.legend()