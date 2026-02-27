"""Clifford Randomised Benchmarking Average Gate Error Metric.

This module provides the Clifford randomised benchmarking average gate
error implementation for the QCMet framework. This metric provides an
estimate of the average gate error of a set of single- and multi-qubit
Clifford gates in a quantum computer. Here the benchmarking procedure
follows M3.3 from arxiv:2502.06717
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

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
        m_list: List[int],
        circs_per_m: int = 5,
        qubits: int | List[int] = 1,
        target_clifford: QuantumCircuit | None = None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the Clifford randomised benchmark.

        Args:
            m_list (List[int]): The list of sequence lengths to run the benchmark on.
            circs_per_m (int): The number of circuits generated for a given sequence length m.
            qubits (int | List[int]): The number of qubits as either a list of qubit
                indices or int specifying number of qubits.
            target_clifford (QuantumCircuit, optional): QuantumCircuit containing only the
                target Clifford gate. This is utilised for Interleaved Clifford Randomised
                Benchmarking. To run Interleaved Clifford Randomised Benchmarking,
                use 'InterleavedRB' Class.
            save_path (str | Path | FileManager | None, optional): Directory path to save results. Defaults to None.

        """
        super().__init__("CliffordRB", qubits=qubits, save_path=save_path)
        self.config["m_list"] = m_list
        self.config["circs_per_m"] = circs_per_m

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
        for m in self.config["m_list"]:
            for _ in range(self.config["circs_per_m"]):
                q_reg = QuantumRegister(self.num_qubits, name="q")
                circ = QuantumCircuit(q_reg)
                # applying clifford gates
                for _ in range(m):
                    circ = circ & random_clifford_circuit(
                        num_qubits=self.num_qubits, num_gates=1
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

    @staticmethod
    def fit_func(m, alpha, a0, b0):
        """Exponential decay fit function.

         This is used for calculating the average gate error.

        Args:
            m (series): sequence length.
            alpha (float): decay fitting parameter.
            a0 (float): amplitude fitting parameter.
            b0 (float): baseline fitting parameter.

        Returns:
            ndarray: fitting function datapoints.

        """
        return a0 * alpha**m + b0

    def _analyze(self):
        """Analyze measurement results average Clifford gate error metric.

        Transforms raw counts into survival probabilities, computes the average survival
        probability for each sequence length m, then calculates the average gate error
        and stores this value in a dictionary.

        Returns:
            dict: {
              'qubits': int,
              'alpha': float,
              'AverageGateError': float
              'fit_results': {'popt': array, 'pcov': array}
            }

        """
        ground_state = "0" * self.num_qubits
        for counts in self._experiment_data["circuit_measurements"].to_list():
            if ground_state in counts.keys():
                self._experiment_data["p_survival"] = self._experiment_data[
                    "circuit_measurements"
                ].apply(
                    lambda x: x.get(ground_state) / self._runtime_params["num_shots"]
                )
                # calculating psurv
            else:
                self._experiment_data["p_survival"] = self._experiment_data[
                    "circuit_measurements"
                ].apply(lambda x: 0)
                # calculating psurv = 0 if no counts in ground state

        av_p_surv_df = (
            self._experiment_data.groupby("m")["p_survival"].mean().reset_index()
        )
        self.p_surv = av_p_surv_df["p_survival"].to_list()

        popt, pcov = curve_fit(
            self.fit_func,
            self.config["m_list"],
            self.p_surv,
            maxfev=20000,
            bounds=[(0, 0, 0), (1, 1, 1)],
        )

        # calculating average gate error
        alpha = popt[0]
        d = 2**self.num_qubits
        self.avg_gate_err = 1 - alpha - (1 - alpha) / d

        self.fit_result = {"fit_result": {"popt": popt, "pcov": pcov}}

        self.run_id = self.file_manager.run_id if self.file_manager else None

        result = {
            "qubits": self.num_qubits,
            "alpha": float(alpha),
            "AverageGateError": self.avg_gate_err,
        } | self.fit_result

        return result

    def _plot(self, axes):
        """Plot survival probability against sequence length.

        Plot of survival probabilities and the fitted exponential decay function.

        Args:
            axes (matplotlib.axes.Axes): Axes to draw the plots on.

        Returns:
            matplotlib.legend.Legend: Legend for the plot.

        """
        if self.target_clifford is None:
            colour = "black"
            type = "RB"
        elif self.target_clifford is not None:
            colour = "green"
            type = "IRB"

        axes.plot(
            self.config["m_list"],
            self.p_surv,
            linestyle="",
            marker="x",
            c=colour,
            label=f"{self._runtime_params['device'].name} {type} results",
        )
        fit_xxs = np.linspace(0, (max(self.config["m_list"])) + 1, 1000)

        axes.plot(
            fit_xxs,
            self.fit_func(fit_xxs, *self.fit_result["fit_result"]["popt"]),
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
