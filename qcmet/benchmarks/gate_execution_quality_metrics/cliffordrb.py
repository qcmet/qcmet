"""Clifford Randomised Benchmarking Average Gate Error Metric.

This module provides the Clifford randomised benchmarking average gate
error implementation for the QCMet framework. This metric provides an
estimate of the average gate error of a set of single- and multi-qubit
Clifford gates in a quantum computer. Here the benchmarking procedure
follows M3.3 from arxiv:2502.06717
"""

from __future__ import annotations

import warnings
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
from scipy.optimize import curve_fit

from qcmet.benchmarks import BaseBenchmark


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

    def _get_fit_model(self):
        """Return the RB decay model, parameter names, bounds, and initial guesses.

        General model:
            p_survival(m) = a0 * alpha**m + b0

        If fixed_a0=True:
            The model is constrained so that p_survival(0) = 1.
            Therefore a0 = 1 - b0.

        If fixed_b0=True:
            b0 = 1 / 2**num_qubits.
        """
        fixed_a0 = self.config.get("fixed_a0", False)
        fixed_b0 = self.config.get("fixed_b0", False)

        b0_fixed = 1 / 2**self.num_qubits

        alpha_guess = 0.999
        a0_guess = 1 - b0_fixed
        b0_guess = b0_fixed

        if fixed_a0 and fixed_b0:
            # Both a0 and b0 are fixed:
            #
            #   b0 = b0_fixed
            #   a0 = 1 - b0_fixed
            #
            # Only alpha is fitted.
            def fit_func(m, alpha):
                return (1 - b0_fixed) * alpha**m + b0_fixed

            param_names = ["alpha"]
            bounds = ([0.0], [1.0])
            p0 = [alpha_guess]

        elif fixed_a0 and not fixed_b0:
            # a0 is constrained by p_survival(0) = 1:
            #
            #   a0 = 1 - b0
            #
            # alpha and b0 are fitted.
            def fit_func(m, alpha, b0):
                return (1 - b0) * alpha**m + b0

            param_names = ["alpha", "b0"]
            bounds = ([0.0, 0.0], [1.0, 1.0])
            p0 = [alpha_guess, b0_guess]

        elif not fixed_a0 and fixed_b0:
            # b0 is fixed:
            #
            #   b0 = b0_fixed
            #
            # alpha and a0 are fitted.
            def fit_func(m, alpha, a0):
                return a0 * alpha**m + b0_fixed

            param_names = ["alpha", "a0"]
            bounds = ([0.0, 0.0], [1.0, 1.0])
            p0 = [alpha_guess, a0_guess]

        else:
            # Fully free model:
            #
            #   p_survival(m) = a0 * alpha**m + b0
            #
            # alpha, a0, and b0 are all fitted.
            def fit_func(m, alpha, a0, b0):
                return a0 * alpha**m + b0

            param_names = ["alpha", "a0", "b0"]
            bounds = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            p0 = [alpha_guess, a0_guess, b0_guess]

        return fit_func, param_names, bounds, p0

    def _fit_log_linear(self, m_values, p_surv, b0_for_log=None):
        """Fit the RB decay using a log-linear transformation.

        The log-linear transform is:

            log(p_survival - b0) = log(a0) + m * log(alpha)

        Args:
            m_values: Sequence lengths.
            p_surv: Survival probabilities.
            b0_for_log: Baseline value to use in the log transform. If None,
                this method requires fixed_b0=True and uses the ideal value
                1 / 2**num_qubits.

        Returns:
            tuple: popt, pcov, param_names, fit_params

        """
        fixed_a0 = self.config.get("fixed_a0", False)
        fixed_b0 = self.config.get("fixed_b0", False)

        if b0_for_log is None:
            if not fixed_b0:
                raise ValueError(
                    "Log-linear fitting requires a known b0. Either set "
                    "fixed_b0=True or pass b0_for_log explicitly."
                )
            b0_for_log = 1 / 2**self.num_qubits

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

        if fixed_a0:
            # Enforce p(0) = 1 using the baseline assumed for this log fit:
            #
            #   a0 = 1 - b0_for_log
            #
            # Therefore:
            #
            #   log_y - log(a0) = m * log(alpha)
            a0_for_log = 1 - b0_for_log
            log_a0 = np.log(a0_for_log)

            y = log_y - log_a0
            x = m_fit

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
            # p_survival = a0 * alpha**m + b0
            #
            # After subtracting the assumed fixed b0:
            #
            #   log_y = log(a0) + m * log(alpha)
            if len(m_fit) < 3:
                raise ValueError(
                    "Need at least three valid data points to estimate both "
                    "alpha and a0, and their covariance, with log-linear fitting."
                )

            coeffs, cov = np.polyfit(m_fit, log_y, deg=1, cov=True)

            log_alpha = coeffs[0]
            log_a0 = coeffs[1]

            alpha = np.exp(log_alpha)
            a0 = np.exp(log_a0)

            # Convert covariance from [log_alpha, log_a0] to [alpha, a0].
            jacobian = np.diag([alpha, a0])
            pcov = jacobian @ cov @ jacobian.T

            popt = np.array([alpha, a0])
            param_names = ["alpha", "a0"]

        fit_params = dict(zip(param_names, popt))

        return popt, pcov, param_names, fit_params

    def _clip_p0_to_bounds(self, p0, bounds):
        """Clip initial guesses so that they lie within curve_fit bounds."""
        lower, upper = bounds

        p0 = np.asarray(p0, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        return np.clip(p0, lower, upper).tolist()

    def _get_log_linear_initial_guess(self, m_values, p_surv):
        """Generate curve_fit initial guesses using a log-linear pre-fit.

        The first-step log-linear fit always assumes the ideal RB baseline:

            b0 = 1 / 2**num_qubits

        This is true even if b0 is free in the final nonlinear curve_fit model.

        If b0 is free in the final nonlinear fit, its initial guess is set to
        the same ideal baseline value.

        Returns:
            tuple:
                p0 (list[float]): Initial guesses for curve_fit, ordered according
                    to self._param_names.
                log_fit_result (dict): Diagnostic result from the log-linear fit.

        Raises:
            ValueError: If the log-linear fit cannot be performed.

        """
        b0_for_log = 1 / 2**self.num_qubits

        log_popt, log_pcov, log_param_names, log_fit_params = self._fit_log_linear(
            m_values,
            p_surv,
            b0_for_log=b0_for_log,
        )

        p0 = list(self._p0)

        for name, value in zip(log_param_names, log_popt):
            if name in self._param_names:
                index = self._param_names.index(name)
                p0[index] = value

        # If b0 is free in the final curve_fit model, initialise it using the
        # same ideal baseline used in the log-linear pre-fit.
        if "b0" in self._param_names:
            b0_index = self._param_names.index("b0")
            p0[b0_index] = b0_for_log

        p0 = self._clip_p0_to_bounds(p0, self._bounds)

        log_fit_result = {
            "status": "success",
            "popt": log_popt,
            "pcov": log_pcov,
            "param_names": log_param_names,
            "params": log_fit_params,
            "assumed_b0": b0_for_log,
        }

        return p0, log_fit_result

    def _analyze(self):
        """Analyze measurement results average Clifford gate error metric.

        Transforms raw counts into survival probabilities, computes the average survival
        probability for each sequence length m, then calculates the average gate error
        and stores this value in a dictionary.


        Returns:
            dict: {
              'qubits': int,
              'alpha': float,
              'AverageGateError': float,
              'fit_result': {
                  'popt': array,
                  'pcov': array,
                  'param_names': list,
                  'params': dict,
                  'fit_method': str,
                  'final_fit_method': str,
              }
            }

        """
        ground_state = "0" * self.num_qubits

        def survival_probability(counts):
            return counts.get(ground_state, 0) / sum(counts.values())

        self._experiment_data["p_survival"] = self._experiment_data[
            "circuit_measurements"
        ].apply(survival_probability)

        av_p_surv_df = (
            self._experiment_data.groupby("m", as_index=False)["p_survival"]
            .mean()
            .sort_values("m")
        )

        m_values = av_p_surv_df["m"].to_numpy()
        p_surv = av_p_surv_df["p_survival"].to_numpy()

        self.m_values = m_values
        self.p_surv = p_surv

        self._fit_func, self._param_names, self._bounds, self._p0 = (
            self._get_fit_model()
        )

        fit_method = self.config.get("fit_method", "two_step")
        log_fit_result = None

        if fit_method == "log":
            # Pure log-linear fitting. This is only allowed when fixed_b0=True,
            popt, pcov, self._param_names, fit_params = self._fit_log_linear(
                m_values,
                p_surv,
            )
            final_fit_method = "log"

        else:
            p0 = self._p0

            if fit_method == "two_step":
                try:
                    p0, log_fit_result = self._get_log_linear_initial_guess(
                        m_values,
                        p_surv,
                    )
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
                        "assumed_b0": 1 / 2**self.num_qubits,
                    }
                    p0 = self._p0

            popt, pcov = curve_fit(
                self._fit_func,
                m_values,
                p_surv,
                p0=p0,
                maxfev=200000,
                bounds=self._bounds,
            )

            fit_params = dict(zip(self._param_names, popt))
            final_fit_method = "curve_fit"

        alpha = fit_params["alpha"]

        d = 2**self.num_qubits
        self.avg_gate_err = (1 - alpha) * (d - 1) / d

        fit_result = {
            "popt": popt,
            "pcov": pcov,
            "param_names": self._param_names,
            "params": fit_params,
            "fit_method": fit_method,
            "final_fit_method": final_fit_method,
        }

        if log_fit_result is not None:
            fit_result["initial_fit"] = {
                "fit_method": "log",
                **log_fit_result,
            }

        self.fit_result = fit_result

        self.run_id = self.file_manager.run_id if self.file_manager else None

        self.result = {
            "qubits": self.num_qubits,
            "alpha": float(alpha),
            "AverageGateError": self.avg_gate_err,
        } | self.fit_result

        return self.result

    def _plot(self, axes):
        """Plot survival probability against sequence length.

        Plot of survival probabilities and the fitted exponential decay function.

        Args:
            axes (matplotlib.axes.Axes): Axes to draw the plots on.

        Returns:
            matplotlib.legend.Legend: Legend for the plot.

        """
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
        fit_xxs = np.linspace(0, (max(self.config["m_list"])) + 1, 1000)

        axes.plot(
            fit_xxs,
            self._fit_func(fit_xxs, *self.fit_result["popt"]),
            linestyle="--",
            marker="",
            c=colour,
            label="Fitted equation",
        )
        axes.set_xlim((0, max(self.config["m_list"])))
        axes.set_ylim((1 / 2**self.num_qubits - 0.05, 1))
        axes.set_xlabel(r"$m$")
        axes.set_ylabel(r"$p_0$")

        return axes.legend()
