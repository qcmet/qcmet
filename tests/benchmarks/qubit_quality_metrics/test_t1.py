"""Tests for T1 time benchmark."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Delay
from qiskit_aer.noise import RelaxationNoisePass

from qcmet import T1, AerSimulator, NoisySimulator


@pytest.fixture
def t1_instance():
    """Create a pytest fixture for T1 benchmark instance with a small set of idle gate counts.

    Returns:
        T1: An instance of the T1 benchmark class with predefined configuration.

    """
    return T1(num_idle_gates_per_circ=np.array([1, 2, 3]))


def test_generate_circuits(t1_instance):
    """Test that the T1 benchmark correctly generates quantum circuits with the expected structure."""
    circuits = t1_instance._generate_circuits()
    assert len(circuits) == 3
    for _, qc in enumerate(circuits):
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 1
        assert qc.data[0].operation.name == "x"
        assert qc.data[-1].operation.name == "measure"


def test_exp_func():
    """Test the exponential decay function used in T1 fitting."""
    x = np.array([0, 1, 1.5], dtype=np.float128)
    amp = 1.0
    dr = 1.5
    expected = amp * np.exp(-1 * x / dr)
    result = T1.exp_func(x, amp, dr)
    np.testing.assert_allclose(result, expected)


def test_analyze_success(t1_instance):
    """Test the analysis method of the T1 benchmark to ensure it successfully fits data."""
    t1_instance._experiment_data = pd.DataFrame(
        {"meas_prob": [{"1": 0.9}, {"1": 0.6}, {"1": 0.3}]}
    )

    # Bypass measurements_to_probabilities
    t1_instance.measurements_to_probabilities = lambda: None

    result = t1_instance._analyze()
    assert result["success"] is True
    assert "T1 (t/t_[1q_gate])" in result
    assert isinstance(result["T1 (t/t_[1q_gate])"], float)


def test_delay_num_idle_gates_error():
    """Test that ValueError raised when arguments are provided for num_idle_gates_per_circ and delay."""
    with pytest.raises(
        ValueError, match="only specify num_idle_gates_per_circ or delay."
    ):
        T1(num_idle_gates_per_circ=np.arange(0, 10, 1), delay=np.arange(0, 10, 1))


def test_delay():
    """Test that delay returns correct axis and T1 labels."""
    experiment = T1(delay=np.arange(0, 10, 1))
    device = NoisySimulator()
    experiment.generate_circuits()
    experiment.run(device, num_shots=100)
    experiment.analyze()
    fig, ax = plt.subplots()
    experiment.plot(axes=ax)
    assert ax.get_xlabel() == r"Delay ($\mu$s)"
    assert "T1 (\u00b5s)" in experiment.result


def test_qubit_index_not_int():
    """Test that ValueError raised when qubit argument is not an int."""
    with pytest.raises(ValueError, match="Qubits indices must be int"):
        T1(qubit_index=[2])


def test_check_t1():
    """Test that benchmark returns correct value of known metric."""
    t1 = 50e-6
    delay_pass = RelaxationNoisePass([t1], [t1], dt=1e-9, op_types=[Delay])
    experiment = T1(
        delay=[0.5, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100], qubit_index=0
    )
    experiment.generate_circuits()
    circuits_delay_pass = []
    for qc in experiment.experiment_data["circuit"]:
        noise_qc = delay_pass(qc)
        circuits_delay_pass.append(noise_qc)
    experiment.experiment_data["circuit"] = circuits_delay_pass
    experiment.run(device=AerSimulator(seed_simulator=42))
    experiment.analyze()
    assert np.isclose(experiment.result["T1 (\u00b5s)"], t1 * 1e6, rtol=0.05)
