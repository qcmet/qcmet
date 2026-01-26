"""Unit tests for the Hamiltonian simulation benchmark in qcmet."""

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qcmet.benchmarks.well_studied_task_execution_quality_metrics.simulation_1d_fermi_hubbard import (
    Simulation1DFermiHubbard,
)


def test_initialization():
    """Test initialization of Simulation1DFermiHubbard with valid parameters.

    Verifies that the configuration dictionary is correctly populated
    and the number of qubits and sites are computed as expected.
    """
    model = Simulation1DFermiHubbard(
        qubits=4, U=2.0, t=1.0, dt=0.1, initial_state=[0, 2], n_steps=1
    )
    assert model.num_qubits == 4
    assert model.config["n_sites"] == 2
    assert model.config["U"] == 2.0
    assert model.config["t"] == 1.0
    assert model.config["dt"] == 0.1
    assert model.config["initial_state"] == [0, 2]


def test_initial_state_circuit():
    """Test initial state preparation.

    Ensures that the initial state circuit correctly flips the specified qubits
    and matches the expected quantum state.
    """
    model = Simulation1DFermiHubbard(qubits=4, initial_state=[1, 3])
    circuit = model.initial_state
    state = Statevector.from_instruction(circuit)
    expected = Statevector.from_label("1010")
    assert state.equiv(expected)


def test_trotter_step_structure():
    """Test structure of the Trotter step.

    Verifies that the Trotter step adds gates to the circuit and includes
    expected gate types such as unitary (F_swap) and XXPlusYY.
    """
    model = Simulation1DFermiHubbard(qubits=4, U=1.0, t=1.0, dt=0.1)
    circuit = QuantumCircuit(4)
    updated_circuit = model._trotter_step(circuit)

    # Check that the circuit has gates applied
    assert len(updated_circuit.data) > 0

    # Check that F_swap and XX+YY gates are present
    gate_names = [instr.operation.name for instr in updated_circuit.data]
    assert "unitary" in gate_names
    assert "xx_plus_yy" in gate_names


def test_even_qubit_assertion():
    """Test assertion for even number of qubits.

    Ensures that initializing the model with an odd number of qubits
    raises an AssertionError.
    """
    with pytest.raises(AssertionError):
        Simulation1DFermiHubbard(qubits=3)  # Odd number of qubits should raise an error
