"""General VQE benchmarks.

This module implements the base `VQE` class, providing the skeleton to set up VQE
benchmark for the QCMet framework. The resulting metric evaluates how well a quantum
computer can reproduce the energy expectation value for a given Hamiltonian and ansatz

The concrete implementation for the 1D Fermi-Hubbard model follows the benchmarking
procedure M5.1 from arxiv:2502.06717.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pathlib import Path

    import openfermion

    from qcmet.core import FileManager
import numpy as np
from openfermion.linalg import get_sparse_operator
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from qcmet.benchmarks import BaseBenchmark


class VQE(BaseBenchmark):
    """Variational Quantum Eigensolver (VQE) benchmark class.

    This class implements a general VQE bencharking workflow, including estimating
    the ground state energy of a given Hamiltonian using a parameterized ansatz
    and initialization circuits and compares the result to the exact energy expectation
    value.

    Concrete VQE instances should either inherit from this base class and overwrite
    the hamiltonian, _apply_ansatz_circuit and initial_state properties/functions,
    or pass in the hamiltonian (as openfermion.QubitOperator), ansatz (as qiskit circuit),
    and init_circuit (as qiskit circuit) components into the constructor.
    """

    def __init__(
        self,
        VQEName: str,
        qubits: int | List[int],
        hamiltonian: openfermion.QubitOperator | None = None,
        ansatz: QuantumCircuit | None = None,
        init_circuit: QuantumCircuit | None = None,
        n_layers: int = 1,
        seed: int| None = None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the VQE benchmark instance with configuration and circuits.

        Stores the Hamiltonian, ansatz, and initial circuit, if these are not passed
        in, the respective properties/functions should be overwritten in inheriting
        classes. Sets the number of layers for the application of the variational
        ansatz.

        Args:
            VQEName (str): Name identifier for the VQE instance.
            qubits (int | List[int]): List of qubit indices.
            hamiltonian (openfermion.QubitOperator, optional): Hamiltonian to be evaluated.
            ansatz (QuantumCircuit, optional): Single layer of the ansatz circuit.
            init_circuit (QuantumCircuit, optional): Initial state preparation circuit.
            n_layers (int): Number of ansatz layers to apply.
            seed (int, optional): Seed used for initialization of random parameters.
            save_path (str | Path | FileManager | None, optional): Directory path to save results.
                Defaults to None.

        """
        super().__init__(VQEName, qubits, save_path=save_path)
        self.config["n_layers"] = n_layers
        if seed is None:
            seed = np.random.randint(10000000)
        self.config["seed"] = seed
        self._hamiltonian = hamiltonian
        self._ansatz = ansatz
        self._init_circuit = init_circuit
        self._full_ansatz = None

        self.params = ParameterVector("params", 0)
        self._parameters = None

    def _add_parameter(self):
        """Add one parameter to the vector of variational parameters.

        Returns:
            ParameterVector: The parameter vector

        """
        self.params.resize(len(self.params) + 1)
        return self.params[-1]

    @property
    def variational_parameters(self):
        """Get or initialize the variational parameters for the ansatz circuit.

        This property ensures that the ansatz circuit is set up before accessing
        the parameters. If the parameters have not been initialized yet, it
        generates a random vector of appropriate length and stores it.

        Returns:
            np.ndarray: The current variational parameters.

        """
        _ = self.full_ansatz_circuit  # make sure ansatz is set up
        if self._parameters is None:
            generator = np.random.default_rng(self.config["seed"])
            parameters = generator.random(len(self.params))
            self._parameters = parameters
        return self._parameters

    @variational_parameters.setter
    def variational_parameters(self, parameters):
        """Set the variational parameters for the ansatz circuit.

        This method raises an error if experiment data has already been generated,
        to prevent inconsistencies due to late parameter changes.

        Args:
            parameters (np.ndarray): The new variational parameters to assign.

        Raises:
            RuntimeError: If experiment data has already been generated.

        """
        if self._experiment_data is not None:
            raise RuntimeError(
                "Circuits have already been generated, it's too "
                "late to change the variational parameters"
            )

        # set the seed to None to indicate parameters were passed in externally
        self.config["seed"] = None
        self._parameters = parameters

    @property
    def full_ansatz_circuit(self):
        """Accessor for the full ansatz circuit (init + layers).

        Returns:
            QuantumCircuit: The full quantum circuit ansatz.

        """
        if self._full_ansatz is None:
            self._full_ansatz = self.initial_state

            for _ in range(self.config["n_layers"]):
                self._full_ansatz = self._apply_ansatz_circuit(self._full_ansatz)
        return self._full_ansatz

    def _generate_circuits(self):
        """Generate all quantum circuits for energy estimation.

        Applies the ansatz layers to the initial state and constructs measurement circuits
        for each unique Pauli term in the Hamiltonian.

        Returns:
            List[dict]: A list of dictionaries containing circuits and measurement strings.

        """
        self.full_ansatz_circuit.assign_parameters(
            self.variational_parameters, inplace=True
        )
        copied_circ = self.full_ansatz_circuit.copy()
        return self._energy_expectation_circuits(copied_circ)

    def _analyze(self):
        """Analyze the energy difference between device and statevector simulations.

        Returns:
            dict: A dictionary containing the average energy difference per site.

        """
        en_device = self.get_energy()
        en_statevector = self.statevector_energy()
        return {"Absolute difference": np.abs(en_device - en_statevector)}

    def statevector_energy(self):
        """Compute the expectation value of the Hamiltonian using statevector simulation.

        Returns:
            float: Real-valued energy expectation from the simulated statevector.

        """
        state = Statevector.from_int(0, 2**self.num_qubits)
        circuit = self.full_ansatz_circuit.copy().reverse_bits()
        state = state.evolve(circuit)

        sparse_ham = get_sparse_operator(self.hamiltonian, n_qubits=self.num_qubits)
        psi = np.array(state)

        return np.real(np.conj(psi).T @ sparse_ham @ psi)

    def _apply_ansatz_circuit(self, circuit):
        """Apply the ansatz circuit to the given quantum circuit.

        Args:
            circuit (QuantumCircuit): The circuit to which the ansatz is applied.

        Returns:
            QuantumCircuit: The updated circuit with the ansatz applied.

        """
        assert self._ansatz is not None

        return circuit.compose(self._ansatz)

    @property
    def hamiltonian(self):
        """Accessor for the Hamiltonian.

        Returns:
            openfermion.QubitOperator: The Hamiltonian as an OpenFermion qubit operator.

        """
        assert self._hamiltonian is not None
        return self._hamiltonian

    def _get_measurement_string(self, term):
        """Generate a measurement string specifying the measurement basis for each qubit.

        This effectively replaces all Z gates with the identity as these do not
        require a basis transformation, and keep X and Y gates to indicate the sites
        for which basis rotations should be applied. Useful for identifying unique
        measurement circuits.

        Args:
            term (tuple): A tuple representing a Pauli term.

        Returns:
            str: A string of Pauli operators (e.g., 'IXY') for measurement.

        """
        measurement_string = ["I"] * self.num_qubits
        for j in term:
            if j[1] != "Z":
                measurement_string[j[0]] = j[1]

        return "".join(measurement_string)

    def _energy_expectation_circuits(self, circuit):
        """Construct measurement circuits for each Pauli term in the Hamiltonian.

        Applies basis change gates and appends measurement operations. Makes sure
        that the same measurement circuit is only generated once (in particular,
        for multiple different "Z" measurement strings).

        Args:
            circuit (QuantumCircuit): The base circuit to copy and modify.

        Returns:
            List[dict]: A list of circuits with associated measurement strings.

        """
        qc_list = []

        hamiltonian = self.hamiltonian

        measurement_circs = {}

        for term in hamiltonian.terms.keys():
            measurement_string = self._get_measurement_string(term)

            if measurement_string not in measurement_circs:
                measurement_circs[measurement_string] = True
                qc_list.append(
                    {
                        "circuit": circuit.copy(),
                        "measurement_string": measurement_string,
                    }
                )
                for j in term:
                    if j[1] == "Y":
                        qc_list[-1]["circuit"].sdg(j[0])
                        qc_list[-1]["circuit"].h(j[0])
                    elif j[1] == "X":
                        qc_list[-1]["circuit"].h(j[0])

                qc_list[-1]["circuit"].measure_all()

        return qc_list

    def get_energy(self):
        """Compute the expectation value of the Hamiltonian from shot data.

        Uses measurement probabilities and parity to evaluate each Pauli-string
        expectation value.

        Returns:
            complex: The estimated energy value from device measurements.

        """
        result = 0.0j

        self.measurements_to_probabilities()

        ham_terms = self.hamiltonian.terms

        for term in ham_terms.keys():
            measurement_string = self._get_measurement_string(term)

            results = self.experiment_data[
                self.experiment_data["measurement_string"] == measurement_string
            ]["meas_prob"]

            for outcome in results:
                for measurement_string in outcome.keys():
                    parity = 0
                    for el in term:
                        parity += int(measurement_string[el[0]])

                    parity = parity % 2
                    result += (
                        ham_terms[term]
                        * outcome[measurement_string]
                        * (-2 * parity + 1)
                    )

        return result

    @property
    def initial_state(self):
        """Return the initial quantum circuit for state preparation.

        Returns:
            QuantumCircuit: The initial state circuit or a default empty circuit.

        """
        if self._init_circuit is None:
            return QuantumCircuit(self.num_qubits)
        else:
            return self._init_circuit
