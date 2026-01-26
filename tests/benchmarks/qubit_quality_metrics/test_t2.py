"""Tests for T2 coherence time benchmark."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Delay
from qiskit_aer.noise import RelaxationNoisePass

from qcmet import T2, AerSimulator, NoisySimulator


@pytest.fixture
def t2_hahn_instance():
    """Create a pytest fixture for T2 Hahn echo benchmark instance.

    Returns:
        T2: An instance of the T2 benchmark class configured for Hahn echo.

    """
    return T2(method="hahn", num_idle_gates_per_circ=np.array([2, 4, 6]))


@pytest.fixture
def t2_ramsey_instance():
    """Create a pytest fixture for T2 Ramsey benchmark instance.

    Returns:
        T2: An instance of the T2 benchmark class configured for Ramsey.

    """
    return T2(method="ramsey", num_idle_gates_per_circ=np.array([1, 2, 3]))


def test_initialization_hahn():
    """Test that T2 benchmark initializes correctly with Hahn echo method."""
    t2 = T2(method="hahn")
    assert t2.config["method"] == "hahn"
    assert t2.num_qubits == 1
    assert t2.config["detuning_phase"] == np.pi / 100
    assert len(t2.config["num_idle_gates_per_circ"]) > 0


def test_initialization_ramsey():
    """Test that T2 benchmark initializes correctly with Ramsey method."""
    t2 = T2(method="ramsey")
    assert t2.config["method"] == "ramsey"
    assert t2.num_qubits == 1
    assert "detuning_phase" in t2.config
    assert t2.config["detuning_phase"] == np.pi / 100
    assert len(t2.config["num_idle_gates_per_circ"]) > 0


def test_initialization_invalid_method():
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match="method must be either 'ramsey' or 'hahn'"):
        T2(method="invalid")


def test_generate_hahn_circuits(t2_hahn_instance):
    """Test that Hahn echo circuits are generated correctly."""
    circuits = t2_hahn_instance._generate_circuits()
    assert len(circuits) == 3
    for qc in circuits:
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 1
        # Check that circuit contains SX gates
        sx_count = sum(1 for instr in qc.data if instr.operation.name == "sx")
        assert (
            sx_count == 4
        )  # Two SX at start, two in middle (π pulse), one at end... wait
        # Actually: SX - [ID+RZ]^(n/2) - SX - SX - [ID+RZ]^(n/2) - SX
        # That's 4 SX gates total
        assert qc.data[-1].operation.name == "measure"


def test_generate_ramsey_circuits(t2_ramsey_instance):
    """Test that Ramsey circuits are generated correctly."""
    circuits = t2_ramsey_instance._generate_circuits()
    assert len(circuits) == 3
    for qc in circuits:
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 1
        # Check that circuit starts with SX and ends with SX then measure
        assert qc.data[0].operation.name == "sx"
        assert qc.data[-1].operation.name == "measure"
        # Count SX gates (should be 2: initial and final)
        sx_count = sum(1 for instr in qc.data if instr.operation.name == "sx")
        assert sx_count == 2


def test_ramsey_fit_func():
    """Test the damped oscillation function used in Ramsey fitting."""
    x = np.array([0, 1, 2], dtype=np.float64)
    amp = 0.5
    dr = 100.0
    f = 0.01
    phi = 0.0
    b = 0.5
    result = T2.ramsey_fit_func(x, amp, dr, f, phi, b)
    expected = amp * np.exp(-x / dr) * np.cos(2 * np.pi * f * x + phi) + b
    np.testing.assert_allclose(result, expected)


def test_hahn_fit_func():
    """Test the exponential decay function used in Hahn echo fitting."""
    x = np.array([0, 1, 1.5], dtype=np.float64)
    amp = 1.0
    dr = 1.5
    b = 0.0
    expected = amp * np.exp(-1 * x / dr) + b
    result = T2.hahn_fit_func(x, amp, dr, b)
    np.testing.assert_allclose(result, expected)


def test_analyze_hahn_success(t2_hahn_instance):
    """Test the analysis method for Hahn echo to ensure it successfully fits data."""
    # Create mock measurement data with exponential decay
    t2_hahn_instance._experiment_data = pd.DataFrame(
        {"meas_prob": [{"0": 0.95}, {"0": 0.75}, {"0": 0.55}]}
    )

    # Bypass measurements_to_probabilities
    t2_hahn_instance.measurements_to_probabilities = lambda: None

    result = t2_hahn_instance._analyze()
    assert result["success"] is True
    assert "T2 (t/t_[1q_gate])" in result
    assert isinstance(result["T2 (t/t_[1q_gate])"], float)
    assert result["method"] == "hahn"


def test_analyze_ramsey_success(t2_ramsey_instance):
    """Test the analysis method for Ramsey to ensure it successfully fits data."""
    # Create mock measurement data with damped oscillation
    # p_1 should oscillate and decay
    t2_ramsey_instance._experiment_data = pd.DataFrame(
        {"meas_prob": [{"1": 0.5}, {"1": 0.3}, {"1": 0.45}]}
    )

    # Bypass measurements_to_probabilities
    t2_ramsey_instance.measurements_to_probabilities = lambda: None

    result = t2_ramsey_instance._analyze()
    assert result["success"] is True
    assert "T2* (t/t_[1q_gate])" in result
    assert isinstance(result["T2* (t/t_[1q_gate])"], float)
    assert result["method"] == "ramsey"


def test_has_plotting(t2_hahn_instance):
    """Test that T2 benchmark has plotting functionality."""
    assert t2_hahn_instance.has_plotting() is True


def test_custom_detuning_phase():
    """Test that custom detuning phase is set correctly."""
    custom_phase = np.pi / 50
    t2 = T2(method="ramsey", detuning_phase=custom_phase)
    assert t2.config["detuning_phase"] == custom_phase


def test_custom_num_idle_gates():
    """Test that custom idle gate counts are set correctly."""
    custom_gates = np.array([10, 20, 30, 40])
    t2_hahn = T2(method="hahn", num_idle_gates_per_circ=custom_gates)
    np.testing.assert_array_equal(
        t2_hahn.config["num_idle_gates_per_circ"], custom_gates
    )

    t2_ramsey = T2(method="ramsey", num_idle_gates_per_circ=custom_gates)
    np.testing.assert_array_equal(
        t2_ramsey.config["num_idle_gates_per_circ"], custom_gates
    )


def test_generate_circuits_user_facing(t2_hahn_instance):
    """Test the user-facing generate_circuits method."""
    t2_hahn_instance.generate_circuits()
    assert len(t2_hahn_instance.circuits) == 3
    assert isinstance(t2_hahn_instance.experiment_data, pd.DataFrame)
    assert len(t2_hahn_instance.experiment_data) == 3


def test_delay_num_idle_gates_error():
    """Test that ValueError raised when arguments are provided for num_idle_gates_per_circ and delay."""
    with pytest.raises(
        ValueError, match="only specify num_idle_gates_per_circ or delay"
    ):
        T2(
            method="ramsey",
            num_idle_gates_per_circ=np.arange(0, 10, 1),
            delay=np.arange(0, 10, 1),
        )


@pytest.mark.parametrize("method, label", [("ramsey", "T2*"), ("hahn", "T2")])
def test_delay(method, label):
    """Test that delay gates approach gives correct result and axis label."""
    experiment = T2(method=method, delay=np.arange(0, 10, 1))
    device = NoisySimulator()
    experiment.generate_circuits()
    experiment.run(device, num_shots=100)
    experiment.analyze()
    fig, ax = plt.subplots()
    experiment.plot(axes=ax)
    assert ax.get_xlabel() == r"Delay ($\mu$s)"
    assert label + " (\u00b5s)" in experiment.result


@pytest.mark.parametrize(
    "argument, detuning_phase", [(None, np.pi / 10), (np.pi, np.pi)]
)
def test_delay_detuning_phase(argument, detuning_phase):
    """Verify that the default and custom detuning phase is correctly applied with delay gates."""
    experiment = T2(method="ramsey", delay=np.arange(0, 10, 1), detuning_phase=argument)
    device = NoisySimulator()
    experiment.generate_circuits()
    experiment.run(device, num_shots=100)
    experiment.analyze()
    fig, ax = plt.subplots()
    experiment.plot(axes=ax)
    assert experiment.config["detuning_phase"] == detuning_phase


def test_qubit_index_int():
    """Test that ValueError raised when multiple qubit indices given."""
    with pytest.raises(ValueError, match="Qubits indices must be int"):
        T2(qubit_index=[1], method="ramsey", delay=np.arange(0, 10, 1))


def test_fit_func_hahn():
    """hahn_fit_func matches manual formula for given parameters."""
    amp, dr, b = 0.5, 1400, 0.5
    m = np.array([0, 1, 2, 3])
    y = T2.hahn_fit_func(m, amp, dr, b)
    expected = amp * np.exp(m * -1 / dr) + b
    assert np.allclose(y, expected)


def test_fit_func_ramsey():
    """ramsey_fit_func matches manual formula for given parameters."""
    amp, dr, f, phi, b = 0.5, 1400, 0.05, 0, 0.5
    m = np.array([0, 1, 2, 3])
    y = T2.ramsey_fit_func(m, amp, dr, f, phi, b)
    expected = amp * np.exp(m * -1 / dr) * np.cos(2 * np.pi * f * m + phi) + b
    assert np.allclose(y, expected)


@pytest.mark.parametrize("method, t2_label", [("hahn", "T2"), ("ramsey", "T2*")])
def test_check_t2(method, t2_label):
    """Test that benchmark returns correct T2* value of known metric."""
    t2 = 25e-6
    delay_pass = RelaxationNoisePass([t2], [t2], dt=1e-9, op_types=[Delay])
    experiment = T2(delay=np.arange(0, 50, 1), method=method, qubit_index=0)
    experiment.generate_circuits()
    circuits_delay_pass = []
    for qc in experiment.experiment_data["circuit"]:
        noise_qc = delay_pass(qc)
        circuits_delay_pass.append(noise_qc)
    experiment.experiment_data["circuit"] = circuits_delay_pass
    experiment.run(device=AerSimulator(seed_simulator=42))
    experiment.analyze()
    assert np.isclose(experiment.result[t2_label + " (\u00b5s)"], t2 * 1e6, rtol=0.1)

