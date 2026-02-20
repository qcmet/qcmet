"""Quantum Fourier Transform Metric.

This module provides the QFT benchmark implementation for the QCMet
framework. This metric evaluates how well a quantum computer can perform
the quantum Fourier transform (QFT). Here the benchmarking procedure
follows M5.4 from arxiv:2502.06717

TODO: the current implementation generates only one QFT instance. It may
be better to generate n circuits to get an average fidelity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pathlib import Path

    from qcmet.core import FileManager
import numpy as np
from qiskit import QuantumCircuit

from qcmet.benchmarks import BaseBenchmark
from qcmet.utils.fidelities import fidelity, normalized_fidelity


class QFT(BaseBenchmark):
    """Implementation of the Quantum Fourier Transform (QFT) metric.

    This class generates random initial states, applies QFT and inverse
    QFT with intermediate rotations, measures the output, and computes
    the fidelity between the device output distribution, against the
    exact distribution.
    """

    def __init__(
        self,
        qubits: int | List[int],
        save_path: str | Path | FileManager | None = None,
        seed: int = 0,
    ):
        """Initialize the QFT benchmark.

        Args:
            qubits (int | List[int]): The number of qubits as either a list of qubit
                indices or int specifying number of qubits.
            save_path (str | Path | FileManager | None, optional): Directory path to save results.
                Defaults to None.
            seed (int, optional): Random seed for state initialization. Defaults to 0.

        """
        super().__init__("QFT", qubits, save_path)
        self.config["seed"] = seed

    def _qft(self, inverse=False):
        """Construct a Quantum Fourier Transform circuit.

        Args:
            inverse (bool): If True, returns the inverse QFT circuit. Defaults to False.

        Returns:
            QuantumCircuit: The constructed (inverse) QFT circuit on self.num_qubits.

        """
        qc = QuantumCircuit(self.num_qubits)

        for i in range(self.num_qubits):
            qc.h(i)
            for j in range(i + 1, self.num_qubits):
                lam = np.pi * (2.0 ** (i - j))
                qc.cp(lam, j, i)

        for i in range(self.num_qubits // 2):
            qc.swap(i, self.num_qubits - i - 1)

        if inverse:
            qc = qc.inverse()

        return qc

    def _generate_circuits(self):
        """Generate the circuit needed to run the QFT metric.

        Builds a circuit such that the QFT is done on a randomly chosen initial state.
        Then the inverse QFT is done so that the expected output is the input state + 1.
        The circuit is then built with the following steps:
            1. Random X gates for initialization.
            2. Forward QFT.
            3. Phase rotations (Rz).
            4. Inverse QFT.
            5. Measurement on all qubits.


        Returns:
            List[Dict]: Each dict contains:
                'circuit' (QuantumCircuit): The full benchmark circuit.
                'random_initialization' (np.ndarray): Bit array of initial X gates.

        """
        qc = QuantumCircuit(self.num_qubits)
        rng = np.random.default_rng(self.config["seed"])
        random_initialization = rng.integers(0, 2, self.num_qubits)

        for i, flag in enumerate(random_initialization):
            if flag == 1:
                qc.x(i)

        qc = qc & self._qft(inverse=False)

        for i in range(self.num_qubits):
            qc.rz(np.pi / 2**i, i)

        qc = qc & self._qft(inverse=True)
        qc.measure_all()
        return [{"circuit": qc, "random_initialization": random_initialization}]

    @staticmethod
    def convert_binary_keys_to_decimal(dictionary):
        """Convert dictionary keys from binary strings to decimal integers.

        Args:
            dictionary (dict): Mapping of binary-string keys (e.g. '0101') to values.

        Returns:
            dict: Mapping of decimal integer keys to the same values.

        Raises:
            ValueError: If the input is not a dictionary.

        """
        if not isinstance(dictionary, dict):
            raise ValueError("Must provide dictionary")
        new_dict = {}
        for key, val in dictionary.items():
            decimal_key = int(key, 2)
            new_dict[decimal_key] = val

        return new_dict

    def _exact_probs_from_random_initialization(self, rand_initial):
        """Compute the exact probability distribution for a given QFT initialization.

        Args:
            rand_initial (array-like): Sequence of bits representing the prepared state.

        Returns:
            np.ndarray: Probability vector of length 2**num_qubits with a single 1.0
                        at the index corresponding to the bitstring, zeros elsewhere.

        """
        exact_probs = np.zeros(2**self.num_qubits)
        if (
            int("".join(str(i) for i in rand_initial), 2) + 1
        ) % 2**self.num_qubits == 0:
            exact_probs[0] = 1
        else:
            exact_probs[int("".join(str(i) for i in rand_initial), 2) + 1] = 1
        return exact_probs

    def _analyze(self):
        """Analyze measurement results to compute QFT metric.

        Transforms raw counts into measured probabilities, converts binary keys
        to decimal, orders probabilities, computes exact distributions for each
        random initialization, then calculates fidelity and normalized fidelity
        and then stores them in a dictionary.

        Returns:
            dict: QFT metric result

        """
        self.measurements_to_probabilities()
        self.order_meas_probs_by_bitstring_decimal_value()
        self.get_exact_probs()

        self.experiment_data["fidelity"] = self.experiment_data.apply(
            lambda row: fidelity(row["exact_probs"], row["ordered_probs"]), axis=1
        )
        self.experiment_data["normalized_fidelity"] = self.experiment_data.apply(
            lambda row: normalized_fidelity(row["exact_probs"], row["ordered_probs"]),
            axis=1,
        )

        return {
            "fidelity": self.experiment_data["fidelity"].to_list(),
            "normalized_fidelity": self.experiment_data[
                "normalized_fidelity"
            ].to_list(),
        }

    def get_exact_probs(self):
        """Compute theoretical probabilities based on random initializations.

        Applies the `_exact_probs_from_random_initialization` method to each
        entry in the `random_initialization` column of `experiment_data`.
        Stores the resulting exact (ideal) probabilities in a new column
        called `exact_probs`.

        Returns:
            None: The method updates `experiment_data` in place.

        """
        self.experiment_data["exact_probs"] = self.experiment_data[
            "random_initialization"
        ].apply(self._exact_probs_from_random_initialization)

    def order_meas_probs_by_bitstring_decimal_value(self):
        """Order measured probabilities by increasing bitstring decimal value.

        Converts binary measurement outcomes from `meas_prob` into their decimal
        representations. Then constructs a new array `ordered_probs` where each
        index corresponds to the decimal value of a bitstring. Missing keys are
        treated as zero-probability.

        Adds two columns to `experiment_data`:
        - `decimal_probs`: a dict mapping int keys (decimal) to probabilities.
        - `ordered_probs`: a NumPy array of length `2**num_qubits`, ordered by key.

        Returns:
            None: The method modifies `experiment_data` in place.

        """
        self.experiment_data["decimal_probs"] = self.experiment_data["meas_prob"].apply(
            self.convert_binary_keys_to_decimal
        )
        self.experiment_data["ordered_probs"] = self.experiment_data[
            "decimal_probs"
        ].apply(
            lambda x: np.array(
                [x[i] if i in x.keys() else 0 for i in range(2**self.num_qubits)]
            )
        )

    def _plot(self, axes):
        """Plot exact and measured probability distributions.

        Args:
            axes (matplotlib.axes.Axes): Axes to draw the bar charts on.

        Returns:
            matplotlib.legend.Legend: Legend for the plotted bars.

        """
        xlabels = [rf"$\vert{i:04b}\rangle$" for i in range(2**self.num_qubits)]
        x_range = np.arange(0, 2**self.num_qubits, dtype=int)
        axes.bar(
            x_range,
            self.experiment_data["exact_probs"][0],
            label="Exact probabilities",
            edgecolor="black",
            color="none",
        )
        axes.bar(
            x_range,
            self.experiment_data["ordered_probs"][0],
            width=0.5,
            label=f"{self._runtime_params['device'].name} results",
            color="black",
            alpha=0.6,
        )
        axes.set_xticks(x_range, xlabels, rotation=45, fontsize=10)
        axes.set_xlabel(r"$x$")
        axes.set_ylabel(r"$p_x$")
        return axes.legend()
