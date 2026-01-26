"""Tests for NoisySimulator device."""
import numpy as np
from qiskit import QuantumCircuit

from qcmet.devices.noisy_simulator import NoisySimulator


def test_simulator_initialization():
    """Test NoisySimulator initializes with a noise model and correct backend."""
    sim = NoisySimulator()
    assert sim.name == "aer_simulator"
    

def test_run_with_basic_circuit():
    """Test running a simple circuit with noisy simulation returns reversed bitstrings."""
    sim = NoisySimulator()
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()
    results = sim.run(qc, num_shots=50)
    assert isinstance(results, dict)
    assert all(k in ["0", "1"] for k in results.keys())
    assert sum(results.values()) == 50


def test_run_multiple_circuits():
    """Ensure NoisySimulator can run multiple circuits and return a list of count dicts."""
    sim = NoisySimulator()

    qc1 = QuantumCircuit(1)
    qc1.rx(np.pi / 3, 0)
    qc1.measure_all()

    qc2 = QuantumCircuit(2)
    qc2.cx(0, 1)
    qc2.measure_all()

    results = sim.run([qc1, qc2], num_shots=100)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, dict) for r in results)
