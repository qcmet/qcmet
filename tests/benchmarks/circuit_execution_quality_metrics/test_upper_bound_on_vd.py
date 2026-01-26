"""test_upper_bound_on_vd.py.

Unit tests for the UpperBoundOnVD benchmark in qcmet.benchmarks.upper_bound_on_vd.
"""
import unittest

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes import RemoveFinalMeasurements

from qcmet import IdealSimulator, NoisySimulator, UpperBoundOnVD


class ParseTargetCircuitThrowsErrorTestCase(unittest.TestCase):
    """TestCase to verify that parse_target_circuit correctly throws errors."""

    def test_parse_target_circuit_with_unallowed_three_qubit_gate(self):
        """Verify that parse_target_circuit throws an exception if the target circuit has a three-qubit gate."""
        circuit = QuantumCircuit(3)
        circuit.ccx(0, 1, 2)
        vd = UpperBoundOnVD(circuit)
        self.assertRaises(ValueError, vd.parse_target_circuit, circuit)

    def test_parse_target_circuit_with_unallowed_two_qubit_gate(self):
        """Verify that parse_target_circuit throws an exception if the target circuit has a non-CZ two-qubit gate."""
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        vd = UpperBoundOnVD(circuit)
        self.assertRaises(ValueError, vd.parse_target_circuit, circuit)

@pytest.fixture
def upper_bound_on_vd_instance():
    """Fixture to create a target circuit satisfying the quantum AP restriction."""
    target_circuit = QuantumCircuit(4)
    for _ in range(10):
        target_circuit.h(0)
        target_circuit.x(1)
        target_circuit.h(3)
        target_circuit.cz(0, 1)
        target_circuit.cz(2, 3)
        target_circuit.x(2)
        target_circuit.h(2)
        target_circuit.cz(1, 2)
        target_circuit.cz(0, 3)
        target_circuit.x(0)
        target_circuit.cz(1, 3)
    vd = UpperBoundOnVD(target_circuit)
    return vd


def test_parse_target_circuit(upper_bound_on_vd_instance):
    """Verify that parse_target_circuit correctly breaks down the target circuit into cycles."""
    upper_bound_on_vd_instance.parse_target_circuit(upper_bound_on_vd_instance.config["target_circuit"])
    circ_structure = upper_bound_on_vd_instance._target_circuit_gate_existence_in_cycles
    assert circ_structure is not None
    assert len(circ_structure) == 30
    assert circ_structure[0]["1q"] == [True, True, False, True]
    assert circ_structure[0]["2q"][0][:2] == [0, 1]
    assert circ_structure[0]["2q"][1][:2] == [2, 3]
    assert circ_structure[1]["1q"] == [False, False, True, False]
    assert circ_structure[1]["2q"][0][:2] == [1, 2]
    assert circ_structure[1]["2q"][1][:2] == [0, 3]
    assert circ_structure[2]["1q"] == [True, False, False, False]
    assert circ_structure[2]["2q"][0][:2] == [1, 3]


def test_generate_example_target_circuit():
    """Verify that generate_example_target_circuit correctly generates a good target circuit."""
    circuit = UpperBoundOnVD.generate_example_target_circuit(2, 4)
    assert circuit is not None
    vd = UpperBoundOnVD(circuit)
    vd.parse_target_circuit(circuit)
    assert vd._target_circuit_gate_existence_in_cycles is not None


def test_generate_circuits(upper_bound_on_vd_instance):
    """Verify that trap circuits are generated correctly, which should give the zero state as output."""
    circuits = upper_bound_on_vd_instance._generate_circuits()
    assert circuits is not None
    unit_vec = np.zeros(2 ** 4)
    unit_vec[0] = 1.0
    for circuit in circuits:
        circuit = RemoveFinalMeasurements()(circuit)
        for gate in circuit.data:
            assert gate.operation.name in ["id", "x", "y", "z", "h", "s", "sdg", "cz", "barrier", "measure"]
        assert np.allclose(np.abs(Operator(circuit).to_matrix() @ unit_vec), unit_vec)

def test_analyze_with_ideal_simulator(upper_bound_on_vd_instance):
    """Verify that the benchmark gives 0 in the noiseless case."""
    device = IdealSimulator()
    upper_bound_on_vd_instance.generate_circuits()
    upper_bound_on_vd_instance.run(device, num_shots=1000)
    result = upper_bound_on_vd_instance.analyze()
    assert result["fails"] == 0
    assert np.allclose(result["upper_bound_on_vd"], 0.0)

def test_analyze_with_noisy_simulator(upper_bound_on_vd_instance):
    """Verify that the benchmark gives non-0 output in the noisy case."""
    device = NoisySimulator()
    upper_bound_on_vd_instance.generate_circuits()
    upper_bound_on_vd_instance.run(device, num_shots=1000)
    result = upper_bound_on_vd_instance.analyze()
    assert 0 <= result["fails"] <= result["total"]
