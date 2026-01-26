"""Unit tests for the VQE benchmark in QCMet."""

import numpy as np
import pytest
from openfermion.ops import QubitOperator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import qcmet
from qcmet.benchmarks.well_studied_task_execution_quality_metrics.vqe import (
    VQE,
)


@pytest.fixture
def sample_hamiltonian():
    """Fixture to provide a sample Hamiltonian with two Pauli terms."""
    return (
        QubitOperator("Z0 Z1", 1.0)
        + QubitOperator("X0", 0.5)
        + QubitOperator("Y1", 0.5)
        + QubitOperator("X0 X1", 0.5)
        + QubitOperator("Y0 Y1", 0.5)
    )


@pytest.fixture
def sample_ansatz():
    """Fixture to provide a sample ansatz circuit with H and CX gates."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def sample_init_circuit():
    """Fixture to provide a sample initial state circuit with an X gate."""
    qc = QuantumCircuit(2)
    qc.x(0)
    return qc


@pytest.fixture
def vqe_instance(sample_hamiltonian, sample_ansatz, sample_init_circuit):
    """Fixture to create a VQE instance with sample components."""
    return VQE(
        VQEName="TestVQE",
        qubits=2,
        hamiltonian=sample_hamiltonian,
        ansatz=sample_ansatz,
        init_circuit=sample_init_circuit,
        n_layers=1,
    )


def test_initialization(vqe_instance):
    """Test that the VQE instance initializes with correct attributes."""
    assert vqe_instance.num_qubits == 2
    assert vqe_instance.hamiltonian == (
        QubitOperator("Z0 Z1", 1.0)
        + QubitOperator("X0", 0.5)
        + QubitOperator("Y1", 0.5)
        + QubitOperator("X0 X1", 0.5)
        + QubitOperator("Y0 Y1", 0.5)
    )
    assert vqe_instance.config["n_layers"] == 1


def test_initial_state_default():
    """Test that the default initial state is an empty circuit with correct qubit count."""
    vqe = VQE("DefaultInit", 2)
    assert isinstance(vqe.initial_state, QuantumCircuit)
    assert vqe.initial_state.num_qubits == 2


def test_initial_state_custom(sample_init_circuit):
    """Test that a custom initial circuit is correctly returned by the initial_state property."""
    vqe = VQE("CustomInit", 2, init_circuit=sample_init_circuit)
    assert vqe.initial_state == sample_init_circuit


def test_apply_ansatz(vqe_instance, sample_ansatz):
    """Test that the ansatz circuit is correctly composed onto a base circuit."""
    base_circuit = QuantumCircuit(2)
    new_circuit = vqe_instance._apply_ansatz_circuit(base_circuit)
    assert new_circuit.num_qubits == 2
    assert new_circuit.data[-2].operation.name == "h"
    assert new_circuit.data[-1].operation.name == "cx"


def test_measurement_string_generation(vqe_instance):
    """Test that the measurement string is correctly generated from a Pauli term."""
    term = ((0, "X"), (1, "Z"))
    m_string = vqe_instance._get_measurement_string(term)
    assert m_string == "XI"


def test_energy_expectation_circuits(vqe_instance):
    """Test that measurement circuits are generated for each unique Pauli term."""
    circuits = vqe_instance._energy_expectation_circuits(vqe_instance.initial_state)
    assert isinstance(circuits, list)
    assert all("circuit" in c and "measurement_string" in c for c in circuits)


def test_statevector_energy(vqe_instance):
    """Test that the statevector energy computation returns a real-valued float."""
    energy = vqe_instance.statevector_energy()
    assert isinstance(energy, float)


def test_computed_energy(vqe_instance):
    """Tests that the exact energy and the estimated energy for perfect probabilities agree (in limit of infinite shots)."""
    vqe_instance.generate_circuits()

    measurements = []
    for c in vqe_instance.experiment_data["circuit"]:
        state = Statevector.from_int(0, 2**vqe_instance.num_qubits).evolve(
            c.remove_final_measurements(False)
        )
        probs = qcmet.AerSimulator.reverse_bitstrings(state.probabilities_dict())
        measurements.append(probs)

    vqe_instance.experiment_data["circuit_measurements"] = measurements
    vqe_instance._runtime_params = {"num_shots": 1}

    # Compare energies
    measured_energy = vqe_instance.get_energy()
    exact_energy = vqe_instance.statevector_energy()

    assert np.isclose(measured_energy.real, exact_energy, atol=1e-10), (
        f"Measured energy {measured_energy} differs from exact energy {exact_energy}"
    )
