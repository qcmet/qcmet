"""Clifford Randomised Benchmarking Average Gate Error Metric.

This module provides the Clifford randomised benchmarking average gate
error implementation for the QCMet framework. This metric provides an
estimate of the average gate error of a set of single- and multi-qubit
Clifford gates in a quantum computer. Here the benchmarking procedure
follows M3.3 from arxiv:2502.06717
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:
    from pathlib import Path

    from qcmet.core import FileManager
import numpy as np
import qiskit.quantum_info as qi
from qiskit import QiskitError, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.random import random_clifford_circuit
from qiskit.quantum_info import Clifford

from qcmet.benchmarks import BaseBenchmark
from qcmet.benchmarks.gate_execution_quality_metrics.rb_fitting import (
    RBFitConfig,
    average_survival_by_m,
    compute_bootstrap_fit_band,
    fit_curve,
    fit_log_linear,
    get_single_log_initial_guess,
    get_single_rb_fit_model,
    single_rb_fit_errors,
)


class CliffordRB(BaseBenchmark):
    """Implements Clifford Randomised Benchmarking Average Gate Error Metric.

    This class generates circuits with a sequence of Clifford gates,
    measures the output, and computes the average gate error of the
    device.

    """

    def __init__(
        self,
        m_list: Sequence[int],
        circs_per_m: int = 5,
        qubits: int | List[int] = 1,
        seed: int | None = None,
        target_clifford: QuantumCircuit | None = None,
        save_path: str | Path | FileManager | None = None,
        fixed_a0: bool = False,
        fixed_b0: bool = False,
        fit_method: str = "two_step",
        errorbar_method: str = "bootstrap",
        bootstrap_samples: int = 1000,
        bootstrap_seed: int | None = None,
        bootstrap_ci_level: float = 0.95,

    ):
        """Initialize the Clifford randomised benchmark.

        Args:
            m_list (Sequence[int]): The list of sequence lengths to run the benchmark on.
            circs_per_m (int): The number of circuits generated for a given sequence length m.
            qubits (int | List[int]): The number of qubits as either a list of qubit
                indices or int specifying number of qubits.
            seed (int | None): Seed for reproducible random Clifford circuit generation.
                If None, circuits are generated non-deterministically. Defaults to None.
            target_clifford (QuantumCircuit, optional): QuantumCircuit containing only the
                target Clifford gate. This is utilised for Interleaved Clifford Randomised
                Benchmarking. To run Interleaved Clifford Randomised Benchmarking,
                use 'InterleavedRB' Class.
            save_path (str | Path | FileManager | None, optional): Directory path to save results. Defaults to None.
            fixed_a0 (bool): Option to fix the amplitude in the fitting to 1. This assumes no SPAM error. Defaults to False.
            fixed_b0 (bool):  Option to fix the baseline in the fit to 1/2**num_qubits. Defaults to False.
            fit_method (str): Fitting method to use. Options are:
                - "two_step": first perform a log-linear fit using the ideal
                baseline b0 = 1 / 2**num_qubits. This provides initial guesses for the
                    nonlinear fit. Then perform scipy.optimize.curve_fit using the
                    model specified by fixed_a0 and fixed_b0.

                    Importantly, even if fixed_b0=False, the log-linear first step
                    still assumes the ideal b0 only for initialisation. The second
                    step can then fit b0.
                - "curve_fit": use nonlinear curve_fit directly.
                - "log": use only the log-linear fit. This requires fixed_b0=True.
                
                Defaults to "two_step".   
            errorbar_method (str): Method used to estimate uncertainty. Options are:

                - "fit": use the covariance matrix returned by the final fit.
                - "bootstrap": bootstrap resample circuits within each sequence length,
                refit each sample, and estimate uncertainty from the bootstrap distribution.
                - "none": do not estimate error bars.

                Defaults to "fit".
            bootstrap_samples (int): Number of bootstrap resamples used when
                errorbar_method="bootstrap". Defaults to 1000.
            bootstrap_seed (int | None): Seed for reproducible bootstrap resampling.
                If None, the bootstrap is non-deterministic. Defaults to None.
            bootstrap_ci_level (float): Confidence interval for calculating 
                the bootstrapped error bars. Defaults to .95.

        """
        super().__init__("CliffordRB", qubits=qubits, save_path=save_path)
        if fit_method not in {"two_step", "curve_fit", "log"}:
            raise ValueError(
                "fit_method must be one of 'two_step', 'curve_fit', or 'log'."
            )

        if fit_method == "log" and not fixed_b0:
            raise ValueError("fit_method='log' requires fixed_b0=True.")

        self.config["fit_method"] = fit_method
        self.config["m_list"] = m_list
        self.config["circs_per_m"] = circs_per_m
        self.config["fixed_a0"] = fixed_a0
        self.config["fixed_b0"] = fixed_b0
        self.config["seed"] = seed
        self.config["errorbar_method"] = errorbar_method
        self.config["bootstrap_samples"] = bootstrap_samples
        self.config["bootstrap_seed"] = bootstrap_seed
        self.config["bootstrap_ci_level"] = bootstrap_ci_level

        if errorbar_method not in {"fit", "bootstrap", "none"}:
            raise ValueError(
                "errorbar_method must be one of 'fit', 'bootstrap', or 'none'."
            )

        if bootstrap_samples <= 1:
            raise ValueError("bootstrap_samples must be greater than 1.")

        if not 0 < bootstrap_ci_level < 1:
            raise ValueError("bootstrap_ci_level must be between 0 and 1.")

        if target_clifford is not None:
            self.config["target_clifford"] = [
                (gate, count) for gate, count in target_clifford.count_ops().items()
            ]
            try:
                Clifford(target_clifford, validate=True)
            except QiskitError as e:
                raise ValueError("target_clifford is not a valid Clifford gate.") from e

        self.target_clifford = target_clifford

    def _generate_circuits(self):
        """Generate Clifford randomised benchmarking circuits.

        Each circuit is built with the following steps:
            1. Apply a sequence of m randomly selected Clifford gates.
            2. Apply a final gate which is the inverse of all previous Clifford gates.
            3. Measure all qubits.

        This procedure is carried out at each sequence length and repeated ncirc times.

        In Interleaved Clifford Randomized Benchmarking, the key difference is that the
        target Clifford gate is inserted after each randomly selected Clifford gate in the sequence.

        Returns:
            List[Dict]: Each dict contains:
                'circuit' (QuantumCircuit): The full benchmark circuit.

        """
        data = []

        seed = self.config.get("seed", None)
        rng = np.random.default_rng(seed)

        for m in self.config["m_list"]:
            for _ in range(self.config["circs_per_m"]):
                q_reg = QuantumRegister(self.num_qubits, name="q")
                circ = QuantumCircuit(q_reg)
                # applying clifford gates
                for _ in range(m):
                    clifford_seed = int(rng.integers(0, np.iinfo(np.uint32).max))

                    circ = circ & random_clifford_circuit(
                        num_qubits=self.num_qubits,
                        num_gates=1,
                        seed=clifford_seed,
                    )
                    circ.barrier()
                    if self.target_clifford is not None:
                        circ = circ & self.target_clifford
                        circ.barrier()
                # applying inverse
                inv = circ.inverse()
                inv_matrix = qi.Operator(inv)
                inv_gate = UnitaryGate(inv_matrix, label="Inverse")
                circ.unitary(inv_gate, q_reg, label="Inverse")

                circ.measure_all()
                if self.target_clifford is not None:
                    data.append(self._circ_with_metadata_dict(circ, m=m, type="IRB"))
                else:
                    data.append(self._circ_with_metadata_dict(circ, m=m, type="RB"))

        return data

    def _analyze(self):
        config = RBFitConfig(
            num_qubits=self.num_qubits,
            fixed_a0=self.config.get("fixed_a0", False),
            fixed_b0=self.config.get("fixed_b0", False),
            fit_method=self.config.get("fit_method", "two_step"),
            errorbar_method=self.config.get("errorbar_method", "fit"),
            bootstrap_samples=self.config.get("bootstrap_samples", 1000),
            bootstrap_seed=self.config.get("bootstrap_seed", None),
            bootstrap_confidence_level=self.config.get(
                "bootstrap_ci_level", 0.95
            ),
        )

        av_df, self._experiment_data = average_survival_by_m(
            self._experiment_data,
            self.num_qubits,
        )

        m_values = av_df["m"].to_numpy(dtype=float)
        p_surv = av_df["p_survival"].to_numpy(dtype=float)

        self.m_values = m_values
        self.p_surv = p_surv

        fit_func, param_names, bounds, p0 = get_single_rb_fit_model(config)

        self._fit_func = fit_func
        self._param_names = param_names
        self._bounds = bounds
        self._p0 = p0

        if config.fit_method == "log":
            popt, pcov, param_names, fit_params = fit_log_linear(
                m_values,
                p_surv,
                config=config,
            )
            log_fit_result = None
            final_fit_method = "log"

        else:
            popt, pcov, fit_params, log_fit_result = fit_curve(
                xdata=m_values,
                ydata=p_surv,
                fit_func=fit_func,
                param_names=param_names,
                bounds=bounds,
                p0=p0,
                config=config,
                log_initial_guess_func=lambda: get_single_log_initial_guess(
                    m_values,
                    p_surv,
                    config,
                    param_names,
                    bounds,
                    p0,
                ),
            )
            final_fit_method = "curve_fit"

        alpha = float(fit_params["alpha"])
        d = 2**self.num_qubits
        self.avg_gate_err = (1 - alpha) * (d - 1) / d

        if config.errorbar_method == "fit":
            errorbars = single_rb_fit_errors(popt, pcov, param_names, config)
                
        elif config.errorbar_method == "bootstrap":
            errorbars = self._get_bootstrap_errors(
                m_values=m_values,
                config=config,
                fit_func=fit_func,
                param_names=param_names,
                bounds=bounds,
                p0=p0,
            )
        else:
            errorbars = {"method": "none", "status": "skipped"}

        fit_result = {
            "popt": popt,
            "pcov": pcov,
            "param_names": param_names,
            "params": fit_params,
            "fit_method": config.fit_method,
            "final_fit_method": final_fit_method,
            "errorbar_method": config.errorbar_method,
            "errorbars": errorbars,
        }

        if log_fit_result is not None:
            fit_result["initial_fit"] = {
                "fit_method": "log",
                **log_fit_result,
            }

        self.fit_result = {"fit_result": fit_result}

        self.result = {
            "qubits": self.num_qubits,
            "alpha": alpha,
            "alpha_stderr": errorbars.get("alpha_stderr"),
            "AverageGateError": self.avg_gate_err,
            "AverageGateError_stderr": errorbars.get("AverageGateError_stderr"),
        } | self.fit_result

        return self.result

    def _plot(self, axes):
        """Plot RB results with bootstrap fit confidence band."""
        colour = "green" if self.target_clifford is not None else "black"
        rb_type = "IRB" if self.target_clifford is not None else "RB"

        axes.plot(
            self.m_values,
            self.p_surv,
            linestyle="",
            marker="x",
            c=colour,
            label=f"{self._runtime_params['device'].name} {rb_type} results",
        )

        fit_xxs = np.linspace(0, max(self.config["m_list"]) + 1, 300)

        popt = self.fit_result["fit_result"]["popt"]

        # --- central fit ---
        y_fit = self._fit_func(fit_xxs, *popt)

        axes.plot(
            fit_xxs,
            y_fit,
            linestyle="--",
            c=colour,
            label="Fitted equation",
        )

        # --- bootstrap band ---
        errorbars = self.fit_result["fit_result"].get("errorbars", {})

        if (
            self.config.get("errorbar_method") == "bootstrap"
            and errorbars.get("status") == "success"
            and "bootstrap_popt_samples" in errorbars
        ):
            band = compute_bootstrap_fit_band(
                fit_func=self._fit_func,
                fit_xs=fit_xxs,
                popt_samples=errorbars["bootstrap_popt_samples"],
                confidence_level=self.config.get("bootstrap_ci_level", 0.68),
            )

            axes.fill_between(
                fit_xxs,
                band["lower"],
                band["upper"],
                color=colour,
                alpha=0.2,
                label=f"Bootstrap CI {self.config.get('bootstrap_ci_level', 0.68):1%}",
            )

        axes.set_xlim((0, max(self.config["m_list"])))
        axes.set_ylim((1 / 2**self.num_qubits - 0.05, 1))
        axes.set_xlabel(r"$m$")
        axes.set_ylabel(r"$p_0$")

        return axes.legend()

            
    def _get_bootstrap_errors(self, m_values, config, fit_func, param_names, bounds, p0):
        """Estimate RB uncertainty by bootstrapping circuits within each m."""
        n_bootstrap = config.bootstrap_samples
        bootstrap_seed = config.bootstrap_seed
        confidence_level = config.bootstrap_confidence_level

        lower_percentile = 100 * (1 - confidence_level) / 2
        median_percentile = 50.0
        upper_percentile = 100 * (1 + confidence_level) / 2

        rng = np.random.default_rng(bootstrap_seed)

        grouped_survival = {
            m: group["p_survival"].to_numpy(dtype=float)
            for m, group in self._experiment_data.groupby("m")
        }

        bootstrap_popt = []
        bootstrap_alpha = []
        bootstrap_avg_gate_err = []
        errorbars = {}
        failed_fits = 0

        d = 2**self.num_qubits

        for _ in range(n_bootstrap):
            try:
                boot_p_surv = []

                for m in m_values:
                    values = grouped_survival[m]

                    if len(values) == 0:
                        raise ValueError("Empty bootstrap group.")

                    resampled_values = rng.choice(values, size=len(values), replace=True)
                    boot_p_surv.append(np.mean(resampled_values))

                boot_p_surv = np.asarray(boot_p_surv, dtype=float)

                if config.fit_method == "log":
                    popt_i, _pcov_i, names_i, fit_params_i = fit_log_linear(
                        m_values,
                        boot_p_surv,
                        config=config,
                    )
                else:
                    popt_i, _pcov_i, fit_params_i, _log_fit_result_i = fit_curve(
                        xdata=m_values,
                        ydata=boot_p_surv,
                        fit_func=fit_func,
                        param_names=param_names,
                        bounds=bounds,
                        p0=p0,
                        config=config,
                        log_initial_guess_func=lambda p_surv=boot_p_surv : get_single_log_initial_guess(
                            m_values,
                            p_surv,
                            config,
                            param_names,
                            bounds,
                            p0,
                        ),
                    )

                alpha_i = float(fit_params_i["alpha"])
                avg_gate_err_i = (1 - alpha_i) * (d - 1) / d

                bootstrap_popt.append(popt_i)
                bootstrap_alpha.append(alpha_i)
                bootstrap_avg_gate_err.append(avg_gate_err_i)
                errorbars["bootstrap_popt_samples"] = bootstrap_popt

            except Exception:
                failed_fits += 1
                continue

        if len(bootstrap_alpha) < 2:
            return {
                "method": "bootstrap",
                "status": "failed",
                "reason": "Fewer than two bootstrap fits succeeded.",
                "n_bootstrap": n_bootstrap,
                "n_successful": len(bootstrap_alpha),
                "n_failed": failed_fits,
                "confidence_level": confidence_level,
            }

        bootstrap_popt = np.asarray(bootstrap_popt, dtype=float)
        bootstrap_alpha = np.asarray(bootstrap_alpha, dtype=float)
        bootstrap_avg_gate_err = np.asarray(bootstrap_avg_gate_err, dtype=float)

        alpha_ci = np.percentile(
            bootstrap_alpha,
            [lower_percentile, median_percentile, upper_percentile],
        )

        avg_gate_err_ci = np.percentile(
            bootstrap_avg_gate_err,
            [lower_percentile, median_percentile, upper_percentile],
        )


        return {
            "method": "bootstrap",
            "status": "success",
            "n_bootstrap": n_bootstrap,
            "n_successful": len(bootstrap_alpha),
            "n_failed": failed_fits,
            "confidence_level": confidence_level,
         
            "alpha_stderr": float(np.std(bootstrap_alpha, ddof=1)),
            "alpha_ci": [float(x) for x in alpha_ci],
      
            "AverageGateError_stderr": float(np.std(bootstrap_avg_gate_err, ddof=1)),
            "AverageGateError_ci": [float(x) for x in avg_gate_err_ci],
        
            "bootstrap_popt_samples": bootstrap_popt,
        }