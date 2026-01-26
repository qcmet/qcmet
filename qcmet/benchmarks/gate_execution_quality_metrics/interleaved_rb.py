"""Interleaved Clifford Randomised Benchmarking Average Gate Error Metric.

This module provides the Interleaved Clifford randomised benchmarking average gate
error implementation for the QCMet framework. This metric provides an
estimate for the average gate error of a target Clifford gate in a gate set.
Here the benchmarking procedure follows M3.4 from arxiv:2502.06717
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

    This class generates circuits with a sequence of Clifford gates,
    measures the output, and computes the average gate error of a specific Clifford gate.

    """

    def __init__(
        self,
        m_list: List[int],
        target_clifford: QuantumCircuit,
        circs_per_m: int = 5,
        qubits: int | List[int] = 1,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the Interleaved Clifford randomised benchmark.

        CliffordRB(interleaved) and CliffordRB(non-interleaved) experiment instances are constructed.

        Args:
            m_list (list): The list of sequence lengths to run the benchmark on.
            circs_per_m (int): The number of circuits generated for a given sequence length m.
            qubits (int | List[int]): The number of qubits as either a list of qubit
                indices or int specifying number of qubits.
            target_clifford (QuantumCircuit): QuantumCircuit containing only the target Clifford gate.
                use 'InterleavedRB' Class.
            save_path (str | Path | FileManager | None, optional): Directory path to save results. Defaults to None.

        """
        super().__init__("InterleavedRB", qubits=qubits, save_path=save_path)
        self.config["m_list"] = m_list
        self.config["circs_per_m"] = circs_per_m

        if target_clifford is None:
            raise ValueError("target Clifford needs to be specified.")

        self.config["target_clifford"] = [
            (gate, count) for gate, count in target_clifford.count_ops().items()
        ]

        self.rb_experiment = CliffordRB(
            m_list, circs_per_m, qubits, target_clifford=None, save_path=save_path
        )
        self.irb_experiment = CliffordRB(
            m_list, circs_per_m, qubits, target_clifford, save_path
        )

    def _generate_circuits(self):
        """Generate circuits for interleaved and non-interleaved Clifford randomised benchmarking.

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
        rb_circs = self.rb_experiment._generate_circuits()
        irb_circs = self.irb_experiment._generate_circuits()

        self.rb_experiment.experiment_data = rb_circs
        self.irb_experiment.experiment_data = irb_circs

        return rb_circs + irb_circs

    def _analyze(self):
        """Analyze measurement results for target Clifford gate error metric.

        The circuit measurements of the interleaved and standard Clifford randomised benchmarks are
        used to calculate their respective decay parameters (alpha_g and alpha) and determine the
        interleaved gate error.

        Returns:
            dict: {
              'qubits': int,
              'alpha': float,
              'alpha_g: float,
              'AverageGateError': float,
              'InterleavedGateError': float,
              'RB_fit': {'fit_results': {'popt': array, 'pcov': array}},
              'IRB_fit': {'fit_results': {'popt': array, 'pcov': array}}
            }

        """
        self.rb_experiment._runtime_params = self._runtime_params
        self.irb_experiment._runtime_params = self._runtime_params
        self.rb_experiment.load_circuit_measurements(
            self.experiment_data[self.experiment_data["type"] == "RB"][
                "circuit_measurements"
            ].tolist()
        )
        self.irb_experiment.load_circuit_measurements(
            self.experiment_data[self.experiment_data["type"] == "IRB"][
                "circuit_measurements"
            ].tolist()
        )

        self.rb_experiment.analyze()
        self.irb_experiment.analyze()

        # calculating average and interleaved gate error
        alpha = float(self.rb_experiment.result["fit_result"]["popt"][0])
        alpha_g = float(self.irb_experiment.result["fit_result"]["popt"][0])
        d = 2**self.num_qubits
        self.avg_gate_err = 1 - alpha - (1 - alpha) / d
        self.int_gate_err = (d - 1) * (1 - (alpha_g / alpha)) / d

        self.run_id = self.file_manager.run_id if self.file_manager else None

        result = {
            "qubits": self.num_qubits,
            "alpha": alpha,
            "alpha_g": alpha_g,
            "AverageGateError": "{:.5f}".format(self.avg_gate_err),
            "InterleavedGateError": "{:.5f}".format(self.int_gate_err),
            "RB_fit": self.rb_experiment.fit_result,
            "IRB_fit": self.irb_experiment.fit_result,
        }

        return result

    def _plot(self, axes):
        """Plot survival probability against sequence length.

        Plot of survival probabilities and also the fitted exponential decay function
        for standard and interleaved Clifford Randomised benchmarks.

        Args:
            axes (matplotlib.axes.Axes): Axes to draw the plots on.

        Returns:
            matplotlib.legend.Legend: Legend for the plot.

        """
        axes.set_xlim((0, max(self.config["m_list"])))
        axes.set_ylim((1 / 2**self.num_qubits - 0.05, 1))

        self.rb_experiment._plot(axes)
        self.irb_experiment._plot(axes)
