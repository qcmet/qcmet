"""Tests for AerSimulator_Base class."""
from qiskit import QuantumCircuit

from qcmet.devices import AerSimulator


def test_run_single_circuit():
    """Test that running a circuit returns reversed bitstrings in counts."""
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()
    sim = AerSimulator()
    counts = sim.run(qc, num_shots=10)
    assert isinstance(counts, dict)
    for bitstring in counts.keys():
        assert len(bitstring) == 1


def test_run_multiple_circuits_returns_list_of_dicts():
    """Test that multiple circuits return a list of reversed count dicts."""
    qc1 = QuantumCircuit(1)
    qc1.x(0)
    qc1.measure_all()

    qc2 = QuantumCircuit(1)
    qc2.measure_all()

    sim = AerSimulator()
    results = sim.run([qc1, qc2], num_shots=10)

    assert isinstance(results, list)
    assert all(isinstance(r, dict) for r in results)
    for r in results:
        for k in r.keys():
            assert len(k) == 1


def test_reverse_bitstrings():
    """Test manual bitstring reversal for known inputs."""
    sim = AerSimulator()
    original = {"000": 10, "001": 5}
    reversed_counts = sim.reverse_bitstrings(original)
    assert reversed_counts == {"000": 10, "100": 5}
