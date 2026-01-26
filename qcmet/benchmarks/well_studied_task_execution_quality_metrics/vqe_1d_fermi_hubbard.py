"""VQE for the 1D Fermi-Hubbard model metric.

This module implements a specialized VQE benchmark for the 1D Fermi-Hubbard model
using the `VQE1DFermiHubbard` class, which extends the base `VQE` class.

This module provides the VQE benchmark implementation for the QCMet
framework. This metric evaluates how well a quantum computer can reproduce the energy
expectation value for the 1D Hubbard model given a Trotterized Hamiltonian ansatz.
Here the benchmarking procedure follows M5.1 from arxiv:2502.06717
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pathlib import Path

    from qcmet.core import FileManager
import cirq
import numpy as np
import openfermion as of
import qiskit
import qiskit.qasm2
from openfermion import QuadraticHamiltonian, jordan_wigner
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate

from .vqe import VQE


class VQE1DFermiHubbard(VQE):
    """Variational Quantum Eigensolver implementation for the 1D Fermi-Hubbard model.

    This class extends the base VQE class to simulate the ground state energy of
    the 1D Fermi-Hubbard model using a parameterized ansatz and Jordan-Wigner
    transformed Hamiltonian.

    """

    def __init__(
        self,
        qubits: int | List[int],
        U: float = 0.0,
        t: float = 1.0,
        shift_number: bool = False,
        save_path: str | Path | FileManager | None = None,
        **kwargs,
    ):
        """Initialize the VQE1DFermiHubbard instance with model parameters.

        Sets up the number of sites, interaction strength, hopping amplitude,
        and whether to shift the number operator.

        Args:
            qubits (int): Total number of qubits (must be even, representing spin-up and spin-down).
            U (float): On-site interaction strength.
            t (float): Hopping amplitude between neighboring sites.
            shift_number (bool): Whether to shift the number operator to center the interaction.
            save_path (str | Path | FileManager | None, optional): Directory path to save results.
                Defaults to None.
            **kwargs: Additional configuration parameters passed to the base VQE class.

        Raises:
            AssertionError: If the number of qubits is not even.

        """
        super().__init__("VQE1DFermiHubbard", qubits, save_path=save_path, **kwargs)
        assert self.num_qubits % 2 == 0
        self.config["n_sites"] = self.num_qubits // 2
        self.config["U"] = U
        self.config["t"] = t
        self.config["shift_number"] = shift_number

    def _apply_ansatz_circuit(self, circuit):
        """Apply the custom ansatz circuit for the 1D Fermi-Hubbard model.

        The ansatz includes:
        - RZZ gates between spin-up and spin-down qubits at each site.
        - XX+YY gates between neighboring spin-up and spin-down qubits
        in two alternating layers.

        Args:
            circuit (QuantumCircuit): The circuit to which the ansatz is applied.

        Returns:
            QuantumCircuit: The updated circuit with one layer of the ansatz applied.

        """
        n_sites = self.config["n_sites"]

        p = self._add_parameter()

        # uu layer
        for i in range(n_sites):
            circuit.rzz(
                p,
                i,
                i + n_sites,
            )
        circuit.barrier()

        # u_h^1
        for i in range(0, n_sites - 1, 2):
            if n_sites > 1:
                p = self._add_parameter()
                circuit.append(XXPlusYYGate(p), [i, i + 1])
                circuit.append(XXPlusYYGate(p), [i + n_sites, i + 1 + n_sites])
        circuit.barrier()

        # u_h^2
        for i in range(1, n_sites - 1, 2):
            if n_sites > 1:
                p = self._add_parameter()
                circuit.append(XXPlusYYGate(p), [i, i + 1])
                circuit.append(XXPlusYYGate(p), [i + n_sites, i + 1 + n_sites])
        circuit.barrier()

        return circuit

    @property
    def initial_state(self):
        """Prepare the initial Gaussian state corresponding to the non-interacting tight-binding Hamiltonian.

        Uses OpenFermion's Gaussian state preparation and converts the Cirq circuit
        to Qiskit via QASM.

        Returns:
            QuantumCircuit: The initial state circuit in Qiskit format.

        """
        n_sites = self.config["n_sites"]
        t = self.config["t"]

        m = np.zeros((2 * n_sites, 2 * n_sites))

        for i in range(n_sites - 1):
            m[i, i + 1] = -t
            m[i + 1, i] = -t
            si = i + n_sites
            m[si, si + 1] = -t
            m[si + 1, si] = -t

        quadratic_ham = QuadraticHamiltonian(m)

        of_init_circuit = cirq.Circuit(
            of.circuits.prepare_gaussian_state(
                cirq.LineQubit.range(self.num_qubits), quadratic_ham
            )
        )

        circ = QuantumCircuit(self.num_qubits)
        circuit = circ.compose(qiskit.qasm2.loads(cirq.qasm(of_init_circuit)))
        circuit.barrier()

        return circuit

    def _interaction_hamiltonian(self):
        """Construct the interaction part of the Fermi-Hubbard Hamiltonian.

        Adds on-site interaction terms between spin-up and spin-down electrons.
        Optionally shifts the number operator to center the interaction.

        Returns:
            FermionOperator: The interaction Hamiltonian in fermionic form.

        """
        i_hamiltonian = 0

        n_sites = self.config["n_sites"]
        U = self.config["U"]

        # interaction terms
        for i in range(n_sites):
            i_hamiltonian += of.FermionOperator(
                "%d^ %d %d^ %d" % (i, i, i + n_sites, i + n_sites), U
            )
            if self.config["shift_number"]:
                i_hamiltonian -= of.FermionOperator("%d^ %d " % (i, i), U / 2)
                i_hamiltonian -= of.FermionOperator(
                    "%d^ %d " % (i + n_sites, i + n_sites), U / 2
                )
                i_hamiltonian += U / 4.0
        return i_hamiltonian

    def _tight_binding_hamiltonian(self):
        """Construct the tight-binding (hopping) part of the Fermi-Hubbard Hamiltonian.

        Adds hopping terms between neighboring sites for both spin-up and spin-down electrons.

        Returns:
            FermionOperator: The tight-binding Hamiltonian in fermionic form.

        """
        tb_hamiltonian = 0
        n_sites = self.config["n_sites"]

        for i in range(2 * n_sites - 1):
            if i + 1 == n_sites:
                continue
            else:
                tb_hamiltonian += of.FermionOperator(
                    "%d^ %d" % (i, i + 1), -self.config["t"]
                ) + of.FermionOperator("%d^ %d" % (i + 1, i), -self.config["t"])

        return tb_hamiltonian

    @property
    def hamiltonian(self):
        """Return the full Fermi-Hubbard Hamiltonian mapped to qubits via Jordan-Wigner transformation.

        Combines the tight-binding and interaction Hamiltonians.

        Returns:
            openfermion.QubitOperator: The full Hamiltonian in qubit representation.

        """
        return jordan_wigner(
            self._tight_binding_hamiltonian() + self._interaction_hamiltonian()
        )
