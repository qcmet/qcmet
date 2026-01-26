"""Quantum dynamics simulation for the 1D Fermi-Hubbard model metric.

This module implements a specialized HamiltonianSimulation benchmark for the 1D
Hubbard model in the Simulation1DFermiHubbard class, which extends the
`HamiltonianSimulation` base class.

This module provides the Hubbard simulation benchmark implementation for the QCMet
framework. This metric evaluates how well a quantum computer can reproduce the populations
for a Trotterized simulation of the 1D Hubbard model starting from a product state.
Here the benchmarking procedure follows M5.3 from arxiv:2502.06717
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pathlib import Path

    from qcmet.core import FileManager
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Operator

from .hamiltonian_simulation import HamiltonianSimulation


class Simulation1DFermiHubbard(HamiltonianSimulation):
    """Simulates the dynamics of a 1D Hubbard model using Trotter decomposition.

    This class extends the `HamiltonianSimulation` base class to implement time
    evolution for a 1D Hubbard model with configurable interaction strength (U),
    hopping parameter (t), time step (dt), and initial state.

    """

    def __init__(
        self,
        qubits: int | List[int],
        U: float = 0.0,
        t: float = 1.0,
        shift_number: bool = False,
        dt: float = 0.1,
        initial_state: tuple[int, ...] = (0,),
        n_steps: int = 1,
        save_path: str | Path | FileManager | None = None,
        **kwargs,
    ):
        """Initialize the 1D Hubbard model dynamics simulation.

        This constructor sets up the configuration for simulating the time evolution
        of a 1D Hubbard model using a Trotterization. The model consists of
        fermionic sites represented by pairs of qubits, with configurable interaction
        strength and hopping amplitude.

        Args:
            qubits (int | List[int]): The number of qubits as either a list of qubit
                indices or int specifying number of qubits.
            U (float): On-site interaction strength. Default is 0.0.
            t (float): Hopping amplitude. Default is 1.0.
            shift_number (bool): Whether to shift particle number. Default is False.
            dt (float): Time step for Trotter evolution. Default is 0.1.
            initial_state (Tuple): Tuple of qubit indices initialized to |1⟩. Default is [0].
            n_steps (int): The number of Trotter steps which are applied.
            save_path (str | Path | FileManager | None, optional): Directory path to save results.
                Defaults to None.
            **kwargs: Additional keyword arguments passed to the base `HamiltonianSimulation` class.

        Raises:
            AssertionError: If the number of qubits is not even.

        """
        super().__init__(
            "Simulation1DFermiHubbard",
            qubits,
            n_steps=n_steps,
            save_path=save_path,
            **kwargs,
        )
        assert self.num_qubits % 2 == 0
        self.config["n_sites"] = self.num_qubits // 2
        self.config["U"] = U
        self.config["t"] = t
        self.config["shift_number"] = shift_number
        self.config["dt"] = dt
        self.config["initial_state"] = initial_state

    def _trotter_step(self, circuit):
        """Apply a single Trotter step to the given quantum circuit.

        Here we use the decomposition outlined in M5.3 of arXiv:2502.06717 including:
        - On-site interaction terms (U_z)
        - Hopping terms (U_e and U_o) using XX+YY gates
        - Fermionic swap operations to simulate particle exchange

        Args:
            circuit (QuantumCircuit): The quantum circuit to which the Trotter step is applied.

        Returns:
            QuantumCircuit: The updated circuit after applying the Trotter step.

        """
        theta = -2 * self.config["t"] * self.config["dt"]

        # U_z
        for qubit in range(self.num_qubits):
            if self.config["U"] != 0.0:
                circuit.rz(self.config["U"] * self.config["dt"] / 2, qubit)

        for qubit in range(0, self.num_qubits, 2):
            if self.config["U"] != 0.0:
                circuit.rzz(self.config["U"] * self.config["dt"] / 2, qubit, qubit + 1)

            # U_e and U_o
            for _ in range(2):
                for qubit in range(1, self.num_qubits - 1, 2):
                    circuit.append(XXPlusYYGate(theta), [qubit, qubit + 1])
                for qubit in range(0, self.num_qubits, 2):
                    op = Operator(
                        data=np.asarray(
                            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]]
                        )
                    )
                    circuit.unitary(op, [qubit, qubit + 1], label="F_swap")

        return circuit

    @property
    def initial_state(self):
        """Constructs the initial quantum state circuit.

        The initial state is defined by flipping the qubits listed in `self.config["initial_state"]`.

        Returns:
            QuantumCircuit: A quantum circuit representing the initial state.

        """
        initial_circuit = QuantumCircuit(self.num_qubits)
        for i in self.config["initial_state"]:
            if i < self.num_qubits:
                initial_circuit.x(i)
        return initial_circuit
