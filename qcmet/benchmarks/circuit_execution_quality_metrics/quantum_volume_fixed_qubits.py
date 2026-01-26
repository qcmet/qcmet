"""Quantum Volume Fixed Qubits Metric.

This module provides the QuantumVolume implementation for the QCMet
framework for a fixed number of qubits. This metric measures the overall
capabilities of a noisy quantum computer. Here the benchmarking
procedure follows M4.1 from arxiv:2502.06717
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pathlib import Path

    from qcmet.core import FileManager
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from qcmet.benchmarks import BaseBenchmark
from qcmet.utils import compute_ideal_outputs


class QuantumVolumeFixedQubits(BaseBenchmark):
    """Implementation of the Quantum Volume Metric.

    This class generates square circuits with Nq qubits and Nq layers of
    gates, where the layers comprise of an Nq qubit gate which randomly
    changes the order of qubits and a column of Haar-random two-qubit
    gates. The ideal probabilites of all bistrings are determined from
    noiseless simulations and the heavy output probability is calculated
    for each circuit. Finally, the condition for the device achieving a
    quantum volume of 2^Nq is checked.

 
    """

    def __init__(
        self,
        trials: int = 100,
        qubits: int | List[int] = 4,
        save_path: str | Path | FileManager | None = None,
        seed: int | None = None,
    ):
        """Initialize the quantum volume benchmark.

        Args:
            trials (int): The number of circuits to generate.
            qubits (int | List[int]): The number of qubits as either a list of qubit
                indices or int specifying number of qubits.
            seed (int, optional): Set the seed for random number generation.
            save_path (str | Path | FileManager | None, optional): Directory path to save results. Defaults to None.

        """
        super().__init__("QuantumVolumeFixedQubits", qubits=qubits, save_path=save_path)
        self.config["seed"] = seed
        self._rng = np.random.default_rng(seed=self.config["seed"])
        self.config["trials"] = trials

    def _random_complex_matrix(self, n):
        """Generate a random complex matrix with dim n*n.

        Seed for random number generation is used if explictly defined in initialization.

        Args:
            n (int): Dimension of the square matrix.

        """
        return self._rng.standard_normal((n, n)) + 1.0j * self._rng.standard_normal(
            (n, n)
        )

    def _haar_measure(self, n):
        """Generate a random unitary from Haar measure with dim n*n (arXiv:math-ph/0609050).

        Args:
            n (int): The dimension of the random unitary from Haar measure.

        """
        rand_mat = self._random_complex_matrix(n)
        q, r = np.linalg.qr(rand_mat)
        d = np.diagonal(r)
        d_normed = d / np.absolute(d)
        return np.multiply(q, d_normed, q)

    def _apply_swap_layer(self, qc, qubits):
        """Apply a swap layer to qubits.

        This randomly changes the order of qubits. In the instance that the random
        ordering of qubits is equal to the initial ordering, the circuit will not
        illustrate that the swap layer has been applied. Seed for random number
        generation is used if explictly defined in initialization.

        Args:
            qc (QuantumCircuit): The square quantum volume circuit.
            qubits (int): The number of qubits in the square quantum volume circuit.

        """
        permutation_list = self._rng.permutation(qubits)
        init_order = np.arange(qubits)
        for i in range(qubits):
            desired_qubit_at_i = permutation_list[i]
            loc_current_to_swap = i
            loc_desired_to_swap = np.where(init_order == desired_qubit_at_i)[0][0]
            if loc_current_to_swap != loc_desired_to_swap:
                qc.swap(loc_current_to_swap, loc_desired_to_swap)
                init_order[loc_current_to_swap], init_order[loc_desired_to_swap] = (
                    init_order[loc_desired_to_swap],
                    init_order[loc_current_to_swap],
                )

    def _apply_su4_layer(self, qc, qubits):
        """Apply a layer of Haar-random 2-qubit (su4) gates.

        Args:
            qc (QuantumCircuit): The square quantum volume circuit.
            qubits (int): The number of qubits in the square quantum volume circuit.

        """
        even_num_qubits = int(np.floor(qubits / 2) * 2)
        for qubit in range(0, even_num_qubits, 2):
            random_su4_gate = self._haar_measure(4)
            qc.append(UnitaryGate(random_su4_gate, label="su4"), [qubit, qubit + 1])

    def _apply_qv_layer(self, qc, qubits):
        """Apply a qv layer comprising of a swap layer and an su4 layer.

        Args:
            qc (QuantumCircuit): The square quantum volume circuit.
            qubits (int): The number of qubits in the square quantum volume circuit.

        """
        self._apply_swap_layer(qc, qubits)
        self._apply_su4_layer(qc, qubits)

    def _generate_single_qv_circuit(self, qubits):
        """Generate a single quantum volume circuit.

        The ciruit is built with the following steps:
            1. Apply a total of Nq qv layers to the circuit.
            2. Measure all qubits.

        Args:
            qubits (int): The number of qubits in the square quantum volume circuit.

        Returns:
            QuantumCircuit: The constructed square quantum volume circuit.

        """
        qc = QuantumCircuit(qubits)
        for _ in range(qubits):
            self._apply_qv_layer(qc, qubits)
        qc.measure_all()

        return qc

    def _generate_circuits(self):
        """Generate the circuits needed to run the quantum volume metric.

        Each circuit is generated using _generate_single_qv_circuit().

        This procedure is repeated for the number of trials specified in the benchmark.

        Returns:
            List[Dict]: Each dict contains:
                'circuit' (QuantumCircuit): The full benchmark circuit.
                'qubits' (int): The number of qubits in the circuit.
                'id', 'hash': Metadata from BaseBenchmark helper

        """
        data = []
        for _ in range(self.config["trials"]):
            qc = self._generate_single_qv_circuit(qubits=self.num_qubits)
            data.append(self._circ_with_metadata_dict(qc, qubits=self.num_qubits))

        return data

    def _get_heavy_outputs(self, ideal_outputs=None):
        """Compute the median ideal probability and determine the heavy outputs.

        The heavy outputs are the output states whose probabilities of being measured are greater than the median probability.
        The ideal_outputs is an optional argument which is only necessary for purpose of testing.

        Args:
            ideal_outputs (dict, optional): Bitstring outputs and their corresponding noiseless probabilities.

        Returns:
            List[str]: Heavy output bitstrings.

        """
        if ideal_outputs is not None:
            self.ideal_outputs = ideal_outputs
        else:
            pass

        values = list(self.ideal_outputs.values())
        median = statistics.median(values)
        self.heavy_outputs = [
            key for key, value in self.ideal_outputs.items() if value > median
        ]

        return self.heavy_outputs

    def _get_heavy_output_counts(self, counts):
        """Find the counts of the heavy outputs.

        Args:
            counts (dict): Dictionary of counts from circuit execution.

        Returns:
            List[int]: Counts of heavy output bitstrings.

        """
        self.heavy_counts = [counts[key] for key in self.heavy_outputs if key in counts]

        return self.heavy_counts

    def _analyze(self):
        """Analyze measurement results to compute the quantum volume metric for a device running Nq qubit circuits.

        The ideal outputs and ideal heavy counts of the circuit are determined and
        the heavy output probability is calculated. This is repeated for each trial
        in the benchmark. Then, the mean and standard error of the heavy output
        probabilities are calculated. The success criterion is computed and the
        success outcome of the device is determined. The mean, success criterion,
        success outcome and minimum quantum volume are stored in a dictionary.

        Returns:
            dict:{
                'mean' (float): Mean heavy output probability
                'mean-2sigma' (float): Confidence interval of 2 sigma below the mean.
                'outcome' (string): "Pass" if success criterion is met, otherwise "Fail".
                'quantum_volume' (str): >= 2^Nq if success criterion is met, otherwise < 2^Nq.

        """
        self._experiment_data["ideal_outputs"] = None
        self._experiment_data["heavy_outputs"] = None

        for i in range(self.config["trials"]):
            self._experiment_data.at[i, "ideal_outputs"] = compute_ideal_outputs(
                qc=self.experiment_data["circuit"][i]
            )
            self.ideal_outputs = compute_ideal_outputs(
                qc=self.experiment_data["circuit"][i]
            )
            self._experiment_data.at[i, "heavy_outputs"] = self._get_heavy_outputs()
            self._get_heavy_outputs()
            self._get_heavy_output_counts(
                self.experiment_data["circuit_measurements"][i]
            )
            self._experiment_data.at[i, "p_h"] = (
                sum(self.heavy_counts) / self._runtime_params["num_shots"]
            )

        self.ph_mean = self._experiment_data["p_h"].mean()
        std_dev = np.sqrt(self.ph_mean * (1 - self.ph_mean) / self.config["trials"])
        self.lower_bound = self.ph_mean - 2 * std_dev
        self.vq = 2**self.num_qubits
        if self.lower_bound > 2 / 3:
            self.outcome = "Pass"
            self.statement = f">= {self.vq}"

        else:
            self.outcome = "Fail"
            self.statement = f"< {self.vq}"

        result = {
            "mean": float(self.ph_mean),
            "mean-2sigma": float(self.lower_bound),
            "outcome": self.outcome,
            "quantum_volume": self.statement,
        }

        return result

    def _plot(self, axes):
        """Plot histogram of the heavy outputs probabilites.

        Horizontal lines are plotted to illustrate  p_h = 2/3 (magenta), mean p_h (black)
        and success criterion (black, dashed).

        Args:
            axes (matplotlib.axes.Axes): Axes object to draw the histogram on.

        Returns:
            matplotlib.legend.Legend: The axes legend instance.

        """
        axes.hist(
            self._experiment_data["p_h"].to_list(),
            bins=np.arange(
                min(self._experiment_data["p_h"]),
                max(self._experiment_data["p_h"]),
                0.02,
            ),
            label=f"{self._runtime_params['device'].name}",
            orientation="horizontal",
            color="lightgrey",
        )

        axes.axhline(y=2 / 3, linestyle="--", color="black", label="2/3")
        axes.axhline(y=self.ph_mean, color="black", label="Mean")
        axes.axhline(
            y=self.lower_bound, linestyle="--", color="magenta", label=r"2$\sigma$"
        )

        axes.set_xlabel("Occurences")
        axes.set_ylabel(r"$p_h$")

        axes.set_title(rf"Quantum Volume: {self.result['quantum_volume']}")
        return axes.legend()
