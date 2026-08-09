"""Tests for AerSimulator_Base class."""

import pytest
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


def test_run_exact_probabilities_single_deterministic_circuit():
    """Test exact mode returns deterministic expected counts scaled by shots."""
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()

    sim = AerSimulator()
    counts = sim.run(qc, num_shots=10)

    assert isinstance(counts, dict)
    assert counts == {"1": pytest.approx(10.0)}


def test_run_exact_probabilities_single_superposition_circuit():
    """Test exact mode returns noise-free expected counts for a superposition."""
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()

    sim = AerSimulator(exact_probabilities=True)
    counts = sim.run(qc, num_shots=10)

    assert isinstance(counts, dict)
    assert counts["0"] == pytest.approx(5.0)
    assert counts["1"] == pytest.approx(5.0)
    assert sum(counts.values()) == pytest.approx(10.0)


def test_run_exact_probabilities_multiple_circuits_returns_list_of_dicts():
    """Test exact mode returns a list of expected-count dictionaries."""
    qc1 = QuantumCircuit(1)
    qc1.x(0)
    qc1.measure_all()

    qc2 = QuantumCircuit(1)
    qc2.h(0)
    qc2.measure_all()

    sim = AerSimulator(exact_probabilities=True)
    results = sim.run([qc1, qc2], num_shots=10)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, dict) for r in results)

    assert results[0] == {"1": pytest.approx(10.0)}

    assert results[1]["0"] == pytest.approx(5.0)
    assert results[1]["1"] == pytest.approx(5.0)
    assert sum(results[1].values()) == pytest.approx(10.0)


def test_run_exact_probabilities_reverses_bitstrings():
    """Test exact mode preserves the simulator's reversed bitstring convention."""
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.measure_all()

    sim = AerSimulator(exact_probabilities=True)
    counts = sim.run(qc, num_shots=10)

    assert counts == {"10": pytest.approx(10.0)}
