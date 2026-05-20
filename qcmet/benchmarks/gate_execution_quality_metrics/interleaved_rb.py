"""Interleaved Clifford Randomised Benchmarking Average Gate Error Metric.

This module provides the Interleaved Clifford randomised benchmarking average gate
error implementation for the QCMet framework. This metric provides an
estimate for the average gate error of a target Clifford gate in a gate set.
Here the benchmarking procedure follows M3.4 from arxiv:2502.06717.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pathlib import Path

    from qiskit import QuantumCircuit

    from qcmet.core import FileManager

from qcmet.benchmarks import BaseBenchmark
from qcmet.benchmarks.gate_execution_quality_metrics import CliffordRB


class InterleavedRB(BaseBenchmark):
    """Implements Interleaved Clifford Randomised Benchmarking Average Gate Error Metric.

    This class generates both standard Clifford RB circuits and interleaved Clifford
    RB circuits, measures the output, and computes the average gate error of a
    specific target Clifford gate.
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
    ):
        """Initialize the Interleaved Clifford randomised benchmark.

        CliffordRB interleaved and non-interleaved experiment instances are
        constructed internally.

        Args:
            m_list (list): The list of sequence lengths to run the benchmark on.
            target_clifford (QuantumCircuit): QuantumCircuit containing only the
                target Clifford gate.
            circs_per_m (int): The number of circuits generated for a given sequence
                length m. Defaults to 5.
            qubits (int | List[int]): The number of qubits as either a list of qubit
                indices or an int specifying the number of qubits. Defaults to 1.
            seed (int | None): Seed for reproducible random Clifford circuit generation.
                If None, circuits are generated non-deterministically. Defaults to None.
            save_path (str | Path | FileManager | None, optional): Directory path to
                save results. Defaults to None.
            fixed_a0 (bool): If True, constrain the amplitude using p_survival(0) = 1,
                so that a0 = 1 - b0. This assumes no SPAM error. Defaults to False.
            fixed_b0 (bool): If True, fix the baseline in the final fit to
                1 / 2**num_qubits. If False, b0 is fitted in the final nonlinear fit.
                Defaults to False.
            fit_method (str): Fitting method passed to the internal CliffordRB
                experiments. Options are:

                - "two_step":
                    First perform a log-linear fit using the ideal baseline
                    b0 = 1 / 2**num_qubits to generate initial guesses. Then perform
                    scipy.optimize.curve_fit using the model specified by fixed_a0
                    and fixed_b0.

                    If fixed_b0=False, the second nonlinear step fits b0.

                - "curve_fit":
                    Use nonlinear curve_fit directly with default initial guesses.

                - "log":
                    Use only the log-linear fit. This requires fixed_b0=True.

                Defaults to "two_step".

        """
        super().__init__("InterleavedRB", qubits=qubits, save_path=save_path)

        if target_clifford is None:
            raise ValueError("target_clifford needs to be specified.")

        if fit_method not in {"two_step", "curve_fit", "log"}:
            raise ValueError(
                "fit_method must be one of 'two_step', 'curve_fit', or 'log'."
            )

        if fit_method == "log" and not fixed_b0:
            raise ValueError("fit_method='log' requires fixed_b0=True.")

        self.config["m_list"] = m_list
        self.config["seed"] = seed
        self.config["circs_per_m"] = circs_per_m
        self.config["fixed_a0"] = fixed_a0
        self.config["fixed_b0"] = fixed_b0
        self.config["fit_method"] = fit_method
        self.config["target_clifford"] = [
            (gate, count) for gate, count in target_clifford.count_ops().items()
        ]

        self.rb_experiment = CliffordRB(
            m_list=m_list,
            circs_per_m=circs_per_m,
            qubits=qubits,
            seed=seed,
            target_clifford=None,
            save_path=save_path,
            fixed_a0=self.config["fixed_a0"],
            fixed_b0=self.config["fixed_b0"],
            fit_method=self.config["fit_method"],
        )

        self.irb_experiment = CliffordRB(
            m_list=m_list,
            circs_per_m=circs_per_m,
            qubits=qubits,
            seed=seed,
            target_clifford=target_clifford,
            save_path=save_path,
            fixed_a0=self.config["fixed_a0"],
            fixed_b0=self.config["fixed_b0"],
            fit_method=self.config["fit_method"],
        )

    def _generate_circuits(self):
        """Generate circuits for interleaved and non-interleaved Clifford RB.

        Standard RB circuits are generated by self.rb_experiment.
        Interleaved RB circuits are generated by self.irb_experiment.

        Returns:
            List[Dict]: Each dict contains:
                'circuit' (QuantumCircuit): The full benchmark circuit.

        """
        rb_circs = self.rb_experiment._generate_circuits()
        irb_circs = self.irb_experiment._generate_circuits()

        self.rb_experiment._experiment_data = rb_circs
        self.irb_experiment._experiment_data = irb_circs

        return rb_circs + irb_circs

    def _analyze(self):
        """Analyze measurement results for the target Clifford gate error metric.

        The circuit measurements of the interleaved and standard Clifford RB
        experiments are used to calculate their respective decay parameters alpha_g
        and alpha. These are then used to estimate the interleaved gate error.

        Returns:
            dict: {
              'qubits': int,
              'alpha': float,
              'alpha_g': float,
              'AverageGateError': str,
              'InterleavedGateError': str,
              'RB_fit': {'fit_result': {'popt': array, 'pcov': array}},
              'IRB_fit': {'fit_result': {'popt': array, 'pcov': array}}
            }
            
        """
        self.rb_experiment._runtime_params = self._runtime_params
        self.irb_experiment._runtime_params = self._runtime_params

        rb_data = self._experiment_data[self._experiment_data["type"] == "RB"].copy()
        irb_data = self._experiment_data[self._experiment_data["type"] == "IRB"].copy()

        if rb_data.empty:
            raise ValueError("No RB measurement data found in experiment data.")

        if irb_data.empty:
            raise ValueError("No IRB measurement data found in experiment data.")

        self.rb_experiment._experiment_data = rb_data
        self.irb_experiment._experiment_data = irb_data

        self.rb_experiment.analyze()
        self.irb_experiment.analyze()

        alpha = float(self.rb_experiment.result["alpha"])
        alpha_g = float(self.irb_experiment.result["alpha"])

        if alpha == 0:
            raise ValueError(
                "Cannot calculate interleaved gate error because alpha is zero."
            )

        d = 2**self.num_qubits

        self.avg_gate_err = (1 - alpha) * (d - 1) / d
        self.int_gate_err = (d - 1) * (1 - (alpha_g / alpha)) / d

        self.run_id = self.file_manager.run_id if self.file_manager else None

        self.result = {
            "qubits": self.num_qubits,
            "alpha": alpha,
            "alpha_g": alpha_g,
            "AverageGateError": "{:.6f}".format(self.avg_gate_err),
            "InterleavedGateError": "{:.6f}".format(self.int_gate_err),
            "RB_fit": self.rb_experiment.fit_result,
            "IRB_fit": self.irb_experiment.fit_result,
        }

        return self.result

    def _plot(self, axes):
        """Plot survival probability against sequence length.

        Plots the survival probabilities and fitted exponential decay functions for
        both the standard and interleaved Clifford RB experiments.

        Args:
            axes (matplotlib.axes.Axes): Axes to draw the plots on.

        Returns:
            matplotlib.legend.Legend: Legend for the plot.

        """
        axes.set_xlim((0, max(self.config["m_list"])))
        axes.set_ylim((1 / 2**self.num_qubits - 0.05, 1))

        self.rb_experiment._plot(axes)
        self.irb_experiment._plot(axes)

        return axes.legend()
