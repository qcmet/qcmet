"""Unit tests for the 1DFermiHubbardVQE benchmark in QCMet."""

import numpy as np
import pytest
from openfermion.ops import FermionOperator, QubitOperator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import qcmet
from qcmet.benchmarks.well_studied_task_execution_quality_metrics.vqe_1d_fermi_hubbard import (
    VQE1DFermiHubbard,
)


@pytest.fixture
def vqe_hubbard_instance():
    """Fixture to create a VQE instance with sample components."""
    return VQE1DFermiHubbard(qubits=4, U=2.0, n_layers=1, shift_number=True)


def test_initialization(vqe_hubbard_instance):
    """Test that the VQE1DFermiHubbard instance initializes correctly."""
    assert vqe_hubbard_instance.num_qubits == 4
    assert vqe_hubbard_instance.config["n_sites"] == 2
    assert vqe_hubbard_instance.config["U"] == 2.0
    assert vqe_hubbard_instance.config["t"] == 1.0
    assert vqe_hubbard_instance.config["shift_number"] is True


def test_initial_state(vqe_hubbard_instance):
    """Test that the initial state circuit is generated correctly."""
    init_circuit = vqe_hubbard_instance.initial_state
    assert isinstance(init_circuit, QuantumCircuit)
    assert init_circuit.num_qubits == 4


def test_interaction_hamiltonian(vqe_hubbard_instance):
    """Test that the interaction Hamiltonian is constructed correctly."""
    interaction_ham = vqe_hubbard_instance._interaction_hamiltonian()
    assert isinstance(interaction_ham, FermionOperator)
    assert len(interaction_ham.terms) > 0


def test_tight_binding_hamiltonian(vqe_hubbard_instance):
    """Test that the tight-binding Hamiltonian is constructed correctly."""
    tb_ham = vqe_hubbard_instance._tight_binding_hamiltonian()
    assert isinstance(tb_ham, FermionOperator)
    assert len(tb_ham.terms) > 0


def test_full_hamiltonian(vqe_hubbard_instance):
    """Test that the full Hamiltonian is correctly mapped to qubit operators."""
    full_ham = vqe_hubbard_instance.hamiltonian
    assert isinstance(full_ham, QubitOperator)
    assert len(full_ham.terms) > 0


def test_computed_energy(vqe_hubbard_instance):
    """Tests that the exact energy and the estimated energy for perfect probabilities agree (in limit of infinite shots)."""
    vqe_hubbard_instance.generate_circuits()

    measurements = []
    for c in vqe_hubbard_instance.experiment_data["circuit"]:
        state = Statevector.from_int(0, 2**vqe_hubbard_instance.num_qubits).evolve(
            c.remove_final_measurements(False)
        )
        probs = qcmet.AerSimulator.reverse_bitstrings(state.probabilities_dict())
        measurements.append(probs)

    vqe_hubbard_instance.experiment_data["circuit_measurements"] = measurements
    vqe_hubbard_instance._runtime_params = {"num_shots": 1}

    # Compare energies
    measured_energy = vqe_hubbard_instance.get_energy()
    exact_energy = vqe_hubbard_instance.statevector_energy()

    assert np.isclose(measured_energy.real, exact_energy, atol=1e-10), (
        f"Measured energy {measured_energy} differs from exact energy {exact_energy}"
    )


def test_state_preparation():
    """Tests that the initialization circuit gives the exact energy for U = 0."""
    experiment = VQE1DFermiHubbard(4, U=0.0, n_layers=0)

    experiment.generate_circuits()

    measurements = []
    for c in experiment.experiment_data["circuit"]:
        state = Statevector.from_int(0, 2**experiment.num_qubits).evolve(
            c.remove_final_measurements(False)
        )
        probs = qcmet.AerSimulator.reverse_bitstrings(state.probabilities_dict())
        measurements.append(probs)

    experiment.experiment_data["circuit_measurements"] = measurements
    experiment._runtime_params = {"num_shots": 1}

    measured_energy = experiment.get_energy()
    exact_energy = experiment.statevector_energy()

    m = np.zeros((2 * experiment.config["n_sites"], 2 * experiment.config["n_sites"]))

    for i in range(experiment.config["n_sites"] - 1):
        m[i, i + 1] = -experiment.config["t"]
        m[i + 1, i] = -experiment.config["t"]
        si = i + experiment.config["n_sites"]
        m[si, si + 1] = -experiment.config["t"]
        m[si + 1, si] = -experiment.config["t"]

    vals = np.linalg.eigvalsh(m)
    reference_energy = np.sum(vals[vals < 0.0])

    assert np.isclose(measured_energy.real, exact_energy, atol=1e-10), (
        f"Measured energy {measured_energy} differs from exact energy {exact_energy}"
    )

    assert np.isclose(reference_energy, measured_energy, atol=1e-10), (
        f"Energy of preparation circuit {measured_energy} differs from reference energy {reference_energy}"
    )
