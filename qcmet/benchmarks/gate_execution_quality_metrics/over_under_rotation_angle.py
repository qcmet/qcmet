"""Over- or under-rotation angle metric.

This module provides the OverUnderRotationAngle benchmark for QCMet. It follows
the protocol M3.6 from arXiv:2502.06717 to measure coherent rotation errors
on one or more qubits by executing pseudoidentity circuits built from a chosen
gate (default SX). The resulting |0⟩ probabilities vs. repetition
count are fitted to an exponentially decaying cosine to extract the per-gate
rotation error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from qcmet.core import FileManager

import numpy as np
import scipy as sp
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import SXGate
from scipy.optimize import curve_fit

from qcmet.benchmarks import BaseBenchmark


class OverUnderRotationAngle(BaseBenchmark):
    r"""Benchmark to estimate coherent over-/under-rotation on one qubit.

    This benchmark generates a family of pseudoidentity circuits of length m,
    measures the probability of the ideal state vs m, and fits the decay and oscillation
    to extract the rotation-angle error per gate.  The fit model is

        prob_0(m) = a + b\*e^(–i\*decay_rate\*m)·cos(m\*theta_err + phase)

    Attributes:
        config (dict): Holds 'delta_m', 'm_max', 'num_gates_for_id', and 'm_array'.
        experiment_data (pd.DataFrame): Populated by BaseBenchmark after generate_circuits.
        fit_result (dict): Holds 'popt' and 'pcov' from curve_fit.
        fit_overrotation_amount (float): Extracted angle error per rotation gate, in units of π.

    """

    def __init__(
        self,
        qubit_index: int = 0,
        delta_m: int = 20,
        gate: Gate = SXGate,
        m_max: int = 200,
        num_gates_for_id: int = 4,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the OverUnderRotationAngle benchmark.

        Args:
            qubit_index (int): The qubit index for routing. Defaults to 0.
            delta_m (int): Step size in repeat count m between pseudoidentity circuits.
            m_max (int): Maximum number of repeats (inclusive).
            gate (qiskit.circuit.Gate): Gate to measure over-under rotation
            num_gates_for_id (int): Number of gates in each repeated pseudoidentity.
            save_path (str | Path | FileManager | None, optional): Directory path to save results. Defaults to None.

        """
        super().__init__(
            "OverUnderRotationAngle", qubits=[qubit_index], save_path=save_path
        )

        self._check_num_gates_for_id(gate, num_gates_for_id)
        self.config["delta_m"] = delta_m
        self.config["gate"] = gate()
        self.config["m_max"] = m_max
        self.config["num_gates_for_id"] = num_gates_for_id
        self.config["m_array"] = np.arange(0, m_max + 1, delta_m)

    def _check_num_gates_for_id(self, gate, num_gates_for_id):
        """Verify that repeating `gate` `num_gates_for_id` times equals identity.

        Args:
            gate (Gate subclass): gate to test.
            num_gates_for_id (int): repeat count.

        Raises:
            ValueError: if gate_matrix**num_gates_for_id != I.

        """
        gate_matrix = gate().to_matrix()
        pseudo_id = np.linalg.matrix_power(gate_matrix, num_gates_for_id)
        if np.allclose(pseudo_id, np.eye(2**self.num_qubits)):
            pass
        else:
            raise ValueError(
                f"{num_gates_for_id} {gate().name} gates is not a pseudoidentity."
            )

    def _generate_circuits(self):
        """Generate pseudoidentity circuits for m in m_array.

        The circuits here differ from M3.6 from arXiv:2502.06717
        by instead preparing qubits in the equator of the Bloch
        sphere before pseudo-identities. The sign of the rotation
        error is determined by looking at the phase.

        Returns:
            List[dict]: A list of records, each with keys:
                - 'circuit': the QuantumCircuit object
                - 'm': repeat count
                - 'id', 'hash': metadata from BaseBenchmark helper

        """
        data = []

        # Get the angles for rotating the qubit into the right frame

        # Get rotation axis to prepare qubit orthogonally to it
        U = self.config["gate"].to_matrix()
        U = U / np.sqrt(np.linalg.det(U))

        sx = np.array([[0, 1], [1, 0]], complex)
        sy = np.array([[0, -1j], [1j, 0]], complex)
        sz = np.array([[1, 0], [0, -1]], complex)

        n_rotation = np.array(
            [
                np.imag(np.trace(U @ sx)),
                np.imag(np.trace(U @ sy)),
                np.imag(np.trace(U @ sz)),
            ]
        )

        n_rotation = n_rotation / np.linalg.norm(n_rotation)

        # create an orthogonal Bloch vector
        ref = np.array([0, 0, 1])
        if abs(np.dot(n_rotation, ref)) > 0.9:
            ref = np.array([0, 1, 0])
        prep_axis = np.cross(n_rotation, ref)

        prep_axis = prep_axis / np.linalg.norm(prep_axis)

        # preparation gate angles
        theta_prep = np.acos(prep_axis[2])
        phi_prep = np.atan2(prep_axis[1], prep_axis[0])

        # measurement gate angles (measurement axis orthogonal to prep and rotation)
        measurement_axis = np.cross(prep_axis, n_rotation)
        measurement_axis = measurement_axis / np.linalg.norm(measurement_axis)

        # read out basis rotation
        theta_readout = np.acos(measurement_axis[2])
        phi_readout = np.atan2(measurement_axis[1], measurement_axis[0])

        for m in self.config["m_array"]:
            quantum_reg = QuantumRegister(self.num_qubits)
            qc = QuantumCircuit(quantum_reg)

            qc.ry(theta_prep, 0)
            qc.rz(phi_prep, 0)

            # Repeat pseudoidentity m times
            for _ in range(m):
                # Pseudoidentity formed by repeating sx gate 4 times
                for _ in range(self.config["num_gates_for_id"]):
                    qc.append(
                        self.config["gate"], np.arange(0, len(self.qubits)).tolist()
                    )
                    qc.barrier()

            # Measure qubit
            qc.ry(theta_readout, 0)
            qc.rz(phi_readout, 0)

            qc.measure_all()
            data.append(self._circ_with_metadata_dict(qc, m=m))

        return data

    @staticmethod
    def fit_func(m, a, b, decay_rate, theta_err, phase):
        """Model function for fitting prob_0 vs m.

        Args:
            m (ndarray): Repeat counts.
            a (float): Baseline offset.
            b (float): Amplitude scaling.
            decay_rate (float): Exponential decay constant.
            theta_err (float): Rotation angle error per pseudoidentity.
            phase (float): Phase offset of the cosine.

        Returns:
            ndarray: Modeled prob_0 values at each m.

        """
        return b * np.exp(-decay_rate * m) * np.cos(theta_err * m + phase) + a

    def _analyze(self):
        """Analyze measurement data, fit the model, and compute rotation error.

        Expects:
          - self._runtime_params['num_shots']
          - self._experiment_data['circuit_measurements']

        Populates:
          - self.experiment_data['p_0']: probability of measuring '0'
          - self.fit_result, self.fit_overrotation_amount, self.success, self.run_id

        Returns:
            dict: {
              'success': bool,
              'OverUnderRotationAngle': float,
              'fit_result': {'popt': array, 'pcov': array}
            }

        """
        self.experiment_data["p_0"] = self.experiment_data[
            "circuit_measurements"
        ].apply(lambda x: x.get("0", 0) / self._runtime_params["num_shots"])

        try:
            f_xx = np.fft.fftfreq(
                len(self.config["m_array"]),
                (self.config["m_array"][1] - self.config["m_array"][0]),
            )  # assume uniform spacing
            f_yy = abs(np.fft.fft(self.experiment_data["p_0"]))
            guess_freq = abs(
                f_xx[np.argmax(f_yy[1:] + 1)]
            )  # excluding the zero frequency "peak", which is related to offset
            guess_amp = np.std(self.experiment_data["p_0"]) * 2.0**0.5
            guess_offset = np.min([np.mean(self.experiment_data["p_0"]), 0.5])
            guess = np.array(
                [guess_offset, guess_amp, 0.001, 2 * np.pi * guess_freq, np.pi]
            )
            popt, pcov = curve_fit(
                self.fit_func,
                self.config["m_array"],
                self.experiment_data["p_0"],
                maxfev=5000,
                p0=guess,
            )
            self.success = True
        except Exception as e:
            self.success = False
            print("Failed to find over/under rotation")
            raise e

        self.fit_result = {"fit_result": {"popt": popt, "pcov": pcov}}
        fit_xxs = np.linspace(0, self.config["m_max"] + 1, 1000)
        yys = self.fit_func(fit_xxs, *self.fit_result["fit_result"]["popt"])

        sign = np.sign(np.gradient(yys))[0]
        self.fit_overrotation_amount = (
            sign * popt[3] / (self.config["num_gates_for_id"])
        )
        self.run_id = self.file_manager.run_id if self.file_manager else None

        result = {
            "success": self.success,
            "OverUnderRotationAngle": self.fit_overrotation_amount,
        } | self.fit_result

        return result

    def _plot(self, axes):
        """Plot measured p_0 vs m and the fitted curve.

        Args:
            axes (Axes): Matplotlib axes object to draw on.

        Returns:
            Legend: The Axes legend instance.

        """
        axes.plot(
            self.config["m_array"],
            self.experiment_data["p_0"],
            linestyle="",
            marker="x",
            c="black",
            label=f"{self._runtime_params['device'].name}",
        )
        fit_xxs = np.linspace(0, self.config["m_max"] + 1, 1000)
        axes.plot(
            fit_xxs,
            self.fit_func(fit_xxs, *self.fit_result["fit_result"]["popt"]),
            linestyle="--",
            marker="",
            c="black",
            label="Fitted equation",
        )
        axes.set_xlim((0, self.config["m_max"]))
        axes.set_ylim((0, 1))
        axes.set_xlabel(r"$m$")
        axes.set_ylabel(r"$p_0$")

        axes.set_title(
            rf"Fitted over- under-rotation angle = {np.round(self.result['OverUnderRotationAngle'], 3)}$\pi$"
        )
        return axes.legend()
