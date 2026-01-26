"""Unit tests for the Hamiltonian simulation benchmark in qcmet."""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qcmet.benchmarks.well_studied_task_execution_quality_metrics.hamiltonian_simulation import (
    HamiltonianSimulation,
)
from qcmet.utils.fidelities import normalized_fidelity


@pytest.fixture
def simple_circuit():
    """Fixture for a simple 2-qubit entangling circuit used as a Trotter step."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def init_circuit():
    """Fixture for a simple 2-qubit initial state preparation circuit."""
    qc = QuantumCircuit(2)
    qc.x(0)
    return qc


def test_initialization(simple_circuit, init_circuit):
    """Test that the HamiltonianSimulation class initializes correctly with given circuits and parameters."""
    dyn = HamiltonianSimulation(
        simulation_name="test",
        qubits=[0, 1],
        evolution_circuit=simple_circuit,
        init_circuit=init_circuit,
        n_steps=2,
    )
    assert dyn.config["n_steps"] == 2
    assert dyn._evolution_circuit == simple_circuit
    assert dyn._init_circuit == init_circuit


def test_initial_state_default():
    """Test that the default initial state is an empty circuit with correct qubit count."""
    dyn = HamiltonianSimulation(simulation_name="test", qubits=[0, 1])
    assert isinstance(dyn.initial_state, QuantumCircuit)
    assert dyn.initial_state.num_qubits == 2


def test_evolution_circuit_composition(simple_circuit, init_circuit):
    """Test that the evolution circuit is composed correctly with multiple Trotter steps and measurements."""
    dyn = HamiltonianSimulation(
        simulation_name="test",
        qubits=[0, 1],
        evolution_circuit=simple_circuit,
        init_circuit=init_circuit,
        n_steps=2,
    )
    circuit = dyn.evolution_circuit
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


def test_generate_circuits(simple_circuit):
    """Test that _generate_circuits returns a list containing the composed evolution circuit."""
    dyn = HamiltonianSimulation(
        simulation_name="test",
        qubits=[0, 1],
        evolution_circuit=simple_circuit,
        n_steps=1,
    )
    circuits = dyn._generate_circuits()
    assert isinstance(circuits, list)
    assert isinstance(circuits[0], QuantumCircuit)


def test_analyze():
    """Test that _analyze computes normalized fidelity correctly using mocked outputs."""
    test_circ = QuantumCircuit(2)
    test_circ.x(0)
    test_circ.h(1)

    dyn = HamiltonianSimulation(
        simulation_name="test",
        qubits=[0, 1],
        evolution_circuit=test_circ,
        n_steps=1,
    )

    dyn.generate_circuits()

    # Mock experiment data
    dyn.experiment_data["circuit_measurements"] = [{"00": 0.1, "11": 0.4, "10": 0.5}]
    dyn._runtime_params = {"num_shots": 1}

    probs = {"10": 1 / 2, "11": 1 / 2, "00": 0.0, "01": 0.0}

    result = dyn._analyze()

    assert "normalized_fidelity" in result
    assert isinstance(result["normalized_fidelity"], list)
    assert np.isclose(
        result["normalized_fidelity"][0],
        normalized_fidelity(list(probs.values()), [0.5, 0.4, 0.0, 0.1]),
    )


def test_trotter_step(simple_circuit):
    """Test that a single Trotter step is correctly composed onto a base circuit."""
    dyn = HamiltonianSimulation(
        simulation_name="test",
        qubits=[0, 1],
        evolution_circuit=simple_circuit,
        n_steps=1,
    )
    base_circuit = QuantumCircuit(2)
    result = dyn._trotter_step(base_circuit)
    assert isinstance(result, QuantumCircuit)
    assert result.num_qubits == 2
