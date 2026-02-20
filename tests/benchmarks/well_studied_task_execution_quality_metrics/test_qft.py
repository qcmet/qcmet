"""Unit tests for the QFT benchmark in qcmet."""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qcmet import IdealSimulator
from qcmet.benchmarks import QFT


def test_convert_binary_keys_to_decimal_valid():
    """Verify that convert_binary_keys_to_decimal works.
     
    It should maps binary-string keys to integer keys, preserving the associated values.
    """
    mapping = {"00": 0.1, "01": 0.2, "10": 0.3, "11": 0.4}
    expected = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
    result = QFT.convert_binary_keys_to_decimal(mapping)
    assert result == expected


def test_convert_binary_keys_to_decimal_invalid():
    """Ensure convert_binary_keys_to_decimal raises a ValueError when the input is not a dictionary."""
    with pytest.raises(ValueError):
        QFT.convert_binary_keys_to_decimal("not a dict")


def test_exact_probs_from_random_initialization():
    """Confirm that _exact_probs_from_random_initialization is correct.
      
    It shoud produce an array of length 2**num_qubits, with the single '1' 
    at the index matching the input bitstring + 1.
    """
    qft = QFT(qubits=3)
    rand_init = np.array([1, 0, 1], dtype=int)

    probs = qft._exact_probs_from_random_initialization(rand_init)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (2**3,)

    # Only index 6 (int('101',2) + 1) should be 1
    assert np.count_nonzero(probs) == 1
    assert probs[6] == 1 


@pytest.mark.parametrize("qubits", [(2), (3), (4)])
def test_exact_probs_from_random_initialization_max_index(qubits):
    """Confirm that _exact_probs_from_random_initialization is correct for rand_initial with highest index.
      
    It shoud produce an array of length 2**num_qubits, with the single '1' 
    at the index matching the input bitstring + 1.
    """
    qft = QFT(qubits=qubits)
    max_statevector = [1] * qubits

    rand_init = np.array(max_statevector, dtype=int)

    probs = qft._exact_probs_from_random_initialization(rand_init)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (2**qubits,)

    # Only index 6 (int('101',2) + 1) should be 1
    assert np.count_nonzero(probs) == 1
    assert probs[0] == 1 


def test_qft_inverse_round_trip():
    """Verify the inverse QFT circuit matches the inverse of the forward QFT circuit."""
    qft = QFT(qubits=3)
    qc_fwd = qft._qft(inverse=False)
    qc_inv = qft._qft(inverse=True)

    assert qc_inv == qc_fwd.inverse()


def test_generate_circuits_structure_and_measurements():
    """Validate the output of _generate_circuits.

    - Returns a single-entry list with 'circuit' and 'random_initialization'.
    - random_initialization is a numpy array of 0s and 1s of correct length.
    - The circuit is a QuantumCircuit with a measurement on each qubit.
    """
    seed = 42
    qft = QFT(qubits=2, seed=seed)
    circuits = qft._generate_circuits()

    assert isinstance(circuits, list) and len(circuits) == 1
    entry = circuits[0]
    assert set(entry.keys()) == {"circuit", "random_initialization"}

    rand = entry["random_initialization"]
    assert isinstance(rand, np.ndarray)
    assert rand.shape == (2,)
    assert set(np.unique(rand)).issubset({0, 1})

    qc = entry["circuit"]
    assert isinstance(qc, QuantumCircuit)
    ops = qc.count_ops()

    # Expect a 'measure' operation on each qubit
    assert ops.get("measure", 0) == 2

def test_qft_with_ideal_sim():
    """Verify ideal simulator returns fidelity of 1."""
    qft = QFT(4,seed=2)
    qft.generate_circuits()
    dummy_sim = IdealSimulator()
    qft.run(dummy_sim, num_shots = 1000)
    qft.analyze()
    assert qft.result['fidelity'] == [1.0]
    assert qft.result['normalized_fidelity'] == [1.0]