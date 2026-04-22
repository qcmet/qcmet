"""Mirrored circuits average polarization Metric.

This module implements the mirrored circuits average polarization metric for QCMet.
This metric evaluates how well a quantum computer can perform a specific target circuit.
Here the benchmarking procedure follows M4.2 from arxiv:2502.06717.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from qcmet.core import FileManager

import numpy as np
from qiskit import QiskitError, QuantumCircuit
from qiskit.quantum_info import Clifford, StabilizerState, random_clifford
from qiskit.synthesis import synth_clifford_full

from qcmet.benchmarks import BaseBenchmark


class MirroredCircuits(BaseBenchmark):
    """Implementation of the mirrored circuits average polarization metric.

    This class takes in or creates a base circuit of Cliffords, and creates the
    mirrored circuits including random Paulis to suppress accidental error cancellation.
    It returns the average polarization metric measuring the average similarity between
    the exact expected output and the measured output bitstrings.
    """

    def __init__(
        self,
        qubits: int | List[int],
        num_circuits: int,
        m: int | None = None,
        clifford_operators: List[Clifford] | None = None,
        seed: int | None = None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the MirroredCircuits benchmark.

        Args:
            qubits (int | List[int]): The number of qubits as either a list of qubit
                indices or int specifying number of qubits.
            num_circuits (int): The number of mirrored circuits k.
            m (int, optional): The number of Clifford layers used when no list of
                Clifford operators is passed in.
            clifford_operators (List[Clifford], optional): The list of Clifford
                operators to use as the base circuit
            seed (int, optional): Random seed to use for randomisations, defaults to None,
                meaning that it will be defined at random
            save_path (str | Path | FileManager | None, optional): Directory path to save
                results. Defaults to None.

        """
        super().__init__("MirroredCircuits", qubits, save_path)

        self.config["num_circuits"] = num_circuits

        if seed is None:
            seed = np.random.randint(100000000)

        self.config["seed"] = seed

        self.rng = np.random.default_rng(seed)

        if clifford_operators is None:
            if type(m) is not int:
                raise TypeError(
                    """If no list of clifford operators is provided, the number of \
                       layers (m) must be specified as an integer to generate layers \
                       of random Cliffords as base circuit."""
                )
            self.clifford_operators_base_circuit = []
            for _ in range(m):
                random_clifford_operator = random_clifford(
                    self.num_qubits, seed=self.rng
                )
                self.clifford_operators_base_circuit.append(random_clifford_operator)
        else:
            self.clifford_operators_base_circuit = clifford_operators

        self.config["cliffords_base_circuit"] = self.clifford_operators_base_circuit

    def _generate_random_mirrored_circuit(self, input_circuit):
        """Build a randomized mirrored circuit around a given Clifford circuit.

        Args:
            input_circuit (QuantumCircuit) : A valid Clifford circuit to mirror.

        Returns:
            Tuple[QuantumCircuit, Clifford]: The mirrored circuit with measurements
                and its Clifford representation before measurement.

        Raises:
            ValueError: If the input circuit does not match the base Clifford composition.

        """
        total_clifford = self.clifford_operators_base_circuit[0]
        for clifford in self.clifford_operators_base_circuit[1:]:
            total_clifford = total_clifford.compose(clifford)

        if Clifford(input_circuit) != total_clifford:
            raise ValueError(
                """The provided list of Clifford operators do not match with \
                    the input circuit."""
            )

        mirrored_circuit = QuantumCircuit(self.num_qubits)

        # L_0: randomly selected single qubit Clifford gates
        circuit_L0 = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            random_clifford_circ = random_clifford(1, seed=self.rng).to_circuit()
            circuit_L0.compose(random_clifford_circ, qubits=[i], inplace=True)

        mirrored_circuit.compose(circuit_L0, inplace=True)

        mirrored_circuit.barrier()

        # C: directly append input circuit to mirrored_circuit
        mirrored_circuit.compose(input_circuit, inplace=True)

        mirrored_circuit.barrier()

        # Q_0: randomly selected Pauli gates
        single_qubit_pauli_gates: List[Any] = [
            mirrored_circuit.id,
            mirrored_circuit.x,
            mirrored_circuit.y,
            mirrored_circuit.z,
        ]
        for i in range(self.num_qubits):
            random_gate: Callable = self.rng.choice(single_qubit_pauli_gates)
            random_gate(i)

        mirrored_circuit.barrier()

        # C~^{-1}: quasi-inverses of C, where each L~_i^{-1}=Q_i.L_i^{-1} for a random pauli Q_i
        for clifford_Li in reversed(self.clifford_operators_base_circuit):
            inverse_Li = clifford_Li.conjugate().transpose()
            inverse_Li_circuit = synth_clifford_full(inverse_Li, method="greedy")
            mirrored_circuit.compose(inverse_Li_circuit, inplace=True)
            for i in range(self.num_qubits):
                random_gate = self.rng.choice(single_qubit_pauli_gates)
                random_gate(i)

            mirrored_circuit.barrier()

        # L_0^{-1}: inverse
        circuit_L0_inverse = circuit_L0.inverse()
        mirrored_circuit.compose(circuit_L0_inverse, inplace=True)

        mirrored_circuit.measure_all(inplace=True)

        return mirrored_circuit

    def _generate_circuits(self):
        """Return the list of circuits for benchmarking workflow.

        Returns:
            list[QuantumCircuit]: A list containing the mirrored circuits to run.

        """
        base_circuit = QuantumCircuit(self.num_qubits)

        for clifford_operator in self.clifford_operators_base_circuit:
            clifford_circuit = synth_clifford_full(clifford_operator, method="greedy")
            base_circuit.compose(clifford_circuit, inplace=True)
            base_circuit.barrier()

        circs = []

        for _ in range(self.config["num_circuits"]):
            mirrored_circuit = self._generate_random_mirrored_circuit(base_circuit)

            circs.append({"circuit": mirrored_circuit})

        return circs

    def _generate_expected_bitstrings(self):
        """Generate the exact bitstrings for the generated circuits via StabilizerState.

        This function adds a new column to the self.experiment_data dataframe containing the
        exact expected bitstring for each circuit.

        """
        expected_bitstrings = []
        for mirrored_circuit in self.experiment_data["circuit"]:
            circ = mirrored_circuit.remove_final_measurements(False).reverse_bits()
            prob = StabilizerState(circ).probabilities_dict(decimals=2)
            if len(prob.keys()) != 1:
                raise QiskitError(
                    f"Oops, something is wrong. The output of the mirrored circuit\
                        must only be a deterministic bitstring, but it's giving {prob}."
                )
            expected_bitstrings.append(list(prob.keys())[0])

        self.experiment_data["expected_bitstrings"] = expected_bitstrings

    def _analyze(self):
        """Evaluate the average polarization by comparing measured and exact output bitstrings.

        Returns:
            dict: Dictionary containing the average polarization.

        """
        self._generate_expected_bitstrings()

        total_polarization = 0.0

        for i in range(len(self.experiment_data["circuit"])):
            if (
                self.experiment_data["expected_bitstrings"][i]
                not in self.experiment_data["circuit_measurements"][i].keys()
            ):
                continue

            correct_counts = self.experiment_data["circuit_measurements"][i][
                self.experiment_data["expected_bitstrings"][i]
            ]

            polarization = (
                correct_counts / self._runtime_params["num_shots"]
                - 1 / 2**self.num_qubits
            ) / (1 - 1 / 2**self.num_qubits)

            total_polarization += polarization

        return {"J": total_polarization / self.config["num_circuits"]}
