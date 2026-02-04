"""test_cycle_benchmarking.py.

Unit tests for the CycleBenchmarking benchmark in qcmet.benchmarks.cycle_benchmarking.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from qiskit import QuantumCircuit

import qcmet as qcm


@pytest.fixture
def simple_g_layer_1q():
    """Create a simple 1-qubit gate layer for testing."""
    qc = QuantumCircuit(1)
    qc.x(0)
    return qc


@pytest.fixture
def simple_g_layer_2q():
    """Create a simple 2-qubit gate layer for testing."""
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    return qc


def test_qubit_indices_not_match_g_layer(simple_g_layer_2q):
    """Test that ValueError raised when qubits does not match g_layer qubits."""
    with pytest.raises(ValueError, match="number of qubits does not match number of qubits in g_layer"):
        qcm.CycleBenchmarking(simple_g_layer_2q,
        [2, 4],
        qubits=[1],
        full_pauli_subspace=False,
        subspace_size=10,
        seed = 42,
        fidelity_method="ratio", )


def test_qubits_when_specified(simple_g_layer_2q):
    """Check that qubit routing works."""
    cb = qcm.CycleBenchmarking(simple_g_layer_2q,
    [2, 4],
    qubits=[1,5],
    full_pauli_subspace=False,
    subspace_size=10,
    seed = 42,
    fidelity_method="ratio", )
    assert cb.qubits == [1,5]


def test_qubits_not_specified(simple_g_layer_2q):
    """Check that qubits is correct when qubit_indices not specified."""
    cb = qcm.CycleBenchmarking(simple_g_layer_2q,
    [2, 4],
    full_pauli_subspace=False,
    subspace_size=10,
    seed = 42,
    fidelity_method="ratio", )
    assert cb.qubits == [0,1]


def test_pauli_subspace(simple_g_layer_2q):
    """Check if Pauli subspace gives actual fidelity from known noise model."""
    cb = qcm.CycleBenchmarking(
        simple_g_layer_2q,
        [2, 4],
        full_pauli_subspace=False,
        subspace_size=10,
        seed = 42,
        fidelity_method="ratio",
    )
    cb.generate_circuits()
    noisy_sim = qcm.NoisySimulator(
        detuning_amount=0,
        error_1q=0,
        error_2q=0.01,
        overrotation_amount=0,
        num_qubits=2,
        t1=0,
        t2=0,
        seed_simulator=10,
    )
    known_fidelity = 0.99
    cb.run(noisy_sim, num_shots=1024)
    cb_fidelity = cb.analyze()["composite_process_fidelity"]["m=2_to_m=4"]
    np.testing.assert_allclose(desired=known_fidelity, actual=cb_fidelity, rtol=0.0015)


def test_initialization_1q(simple_g_layer_1q):
    """Test that CycleBenchmarking initializes correctly for 1 qubit."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_1q,
        repetitions_list=[1, 2],
        num_random_sequences=2,
        full_pauli_subspace=True,
    )
    assert cb.num_qubits == 1
    assert cb.config["repetitions_list"] == [1, 2]
    assert cb.config["num_random_sequences"] == 2
    assert cb.config["full_pauli_subspace"] is True
    # For 1 qubit: 4^1 - 1 = 3 Pauli operators (X, Y, Z)
    assert len(cb.pauli_list) == 3


def test_initialization_2q(simple_g_layer_2q):
    """Test that CycleBenchmarking initializes correctly for 2 qubits."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_2q,
        repetitions_list=[2, 4],
        num_random_sequences=3,
        full_pauli_subspace=True,
    )
    assert cb.num_qubits == 2
    # For 2 qubits: 4^2 - 1 = 15 Pauli operators
    assert len(cb.pauli_list) == 15


def test_initialization_subspace(simple_g_layer_2q):
    """Test initialization with random Pauli subspace."""
    subspace_size = 5
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_2q,
        repetitions_list=[2, 4],
        num_random_sequences=2,
        full_pauli_subspace=False,
        subspace_size=subspace_size,
    )
    assert len(cb.pauli_list) == subspace_size
    assert cb.config["subspace_size"] == subspace_size


def test_initialization_invalid_method(simple_g_layer_1q):
    """Test that invalid fidelity method raises ValueError."""
    with pytest.raises(ValueError, match="fidelity_method must be either"):
        qcm.CycleBenchmarking(
            g_layer=simple_g_layer_1q,
            repetitions_list=[1, 2],
            fidelity_method="invalid",
        )


def test_initialization_missing_subspace_size(simple_g_layer_2q):
    """Test that missing subspace_size raises ValueError."""
    with pytest.raises(ValueError, match="subspace_size must be specified"):
        qcm.CycleBenchmarking(
            g_layer=simple_g_layer_2q,
            repetitions_list=[2, 4],
            full_pauli_subspace=False,
        )


def test_initialization_invalid_subspace_size(simple_g_layer_1q):
    """Test that too large subspace_size raises ValueError."""
    with pytest.raises(ValueError, match="larger than full subspace"):
        qcm.CycleBenchmarking(
            g_layer=simple_g_layer_1q,
            repetitions_list=[1, 2],
            full_pauli_subspace=False,
            subspace_size=10,  # Max for 1 qubit is 3
        )


def test_pauli_label_generation(simple_g_layer_2q):
    """Test Pauli label generation."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_2q,
        repetitions_list=[2],
        num_random_sequences=1,
        full_pauli_subspace=True,
    )
    # Test a sample Pauli channel
    from qiskit.circuit.library import XGate, YGate

    label = cb._get_pauli_label((XGate, YGate))
    assert label == "XY"


def test_generate_circuits_count(simple_g_layer_1q):
    """Test that correct number of circuits are generated."""
    num_reps = [1, 2]
    num_sequences = 3
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_1q,
        repetitions_list=num_reps,
        num_random_sequences=num_sequences,
        full_pauli_subspace=True,
    )
    cb.generate_circuits()

    # Expected: len(repetitions) * len(pauli_list) * num_sequences
    # For 1 qubit: 2 * 3 * 3 = 18
    expected_circuits = len(num_reps) * len(cb.pauli_list) * num_sequences
    assert len(cb.circuits) == expected_circuits


def test_generate_circuits_structure(simple_g_layer_1q):
    """Test that generated circuits have correct structure."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_1q,
        repetitions_list=[1],
        num_random_sequences=1,
        full_pauli_subspace=True,
    )
    cb.generate_circuits()

    # Check that circuits have measurements
    for circ in cb.circuits:
        assert isinstance(circ, QuantumCircuit)
        assert circ.num_qubits == 1
        # Check that measurement is present
        ops = [instr.operation.name for instr in circ.data]
        assert "measure" in ops


def test_metadata_in_experiment_data(simple_g_layer_1q):
    """Test that experiment data contains required metadata."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_1q,
        repetitions_list=[1, 2],
        num_random_sequences=2,
        full_pauli_subspace=True,
    )
    cb.generate_circuits()

    # Check metadata columns
    assert "m" in cb.experiment_data.columns
    assert "pauli_channel" in cb.experiment_data.columns

    # Check values
    assert set(cb.experiment_data["m"].unique()) == {1, 2}
    assert len(cb.experiment_data["pauli_channel"].unique()) == 3  # X, Y, Z for 1q


def test_pauli_expectation_calculation(simple_g_layer_1q):
    """Test Pauli expectation value calculation."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_1q,
        repetitions_list=[1],
        num_random_sequences=1,
    )

    # Test with known counts
    counts = {"0": 800, "1": 200}
    exp = cb._calculate_pauli_expectation("X", counts, 1000)
    # For X Pauli, parity is -1 for |1⟩, so: (800 - 200) / 1000 = 0.6
    assert exp == 0.6

    # Test with Z Pauli
    exp_z = cb._calculate_pauli_expectation("Z", counts, 1000)
    assert exp_z == 0.6


def test_analyze_ratio_method(simple_g_layer_1q):
    """Test analysis with ratio method."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_1q,
        repetitions_list=[2, 4],
        num_random_sequences=5,
        fidelity_method="ratio",
    )
    cb.generate_circuits()

    # Run on ideal simulator
    ideal_sim = qcm.IdealSimulator()
    cb.run(device=ideal_sim, num_shots=1024)

    result = cb.analyze()

    assert result["success"] is True
    assert result["method"] == "ratio"
    assert "composite_process_fidelity" in result
    assert isinstance(result["composite_process_fidelity"], dict)

    # For ideal simulator, fidelity should be close to 1
    for fid in result["composite_process_fidelity"].values():
        np.testing.assert_allclose(fid, 1.0, 0.01)  # Allow some tolerance


def test_analyze_fit_method(simple_g_layer_1q):
    """Test analysis with fit method."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_1q,
        repetitions_list=[2, 4, 6],
        num_random_sequences=5,
        fidelity_method="fit",
    )
    cb.generate_circuits()

    # Run on ideal simulator
    ideal_sim = qcm.IdealSimulator()
    cb.run(device=ideal_sim, num_shots=1024)

    result = cb.analyze()

    assert result["success"] is True
    assert result["method"] == "fit"
    assert "composite_process_fidelity" in result
    assert isinstance(result["composite_process_fidelity"], float)
    assert "fit_params" in result

    # For ideal simulator, fidelity should be close to 1
    np.testing.assert_allclose(result["composite_process_fidelity"], 1.0, 0.01)


def test_analyze_with_noisy_simulator(simple_g_layer_2q):
    """Test that noisy simulator produces lower fidelity."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_2q,
        repetitions_list=[2, 4, 8],
        num_random_sequences=20,
        fidelity_method="fit",
        # full_pauli_subspace=False,
        # subspace_size=5,  # Use smaller subspace for faster testing
    )
    cb.generate_circuits()

    # Run on noisy simulator
    noisy_sim = qcm.NoisySimulator(
        overrotation_amount=0,
        detuning_amount=0,
        error_1q=0.0,
        error_2q=0.08,
        t1=0,
        t2=0,
    )
    cb.run(device=noisy_sim, num_shots=1024)

    result = cb.analyze()
    assert result["success"] is True
    # Noisy simulator should produce fidelity < 1
    np.testing.assert_allclose(result["composite_process_fidelity"], 0.92, 0.01)


def test_cycle_fidelities_in_result(simple_g_layer_1q):
    """Test that cycle fidelities are stored in result."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_1q,
        repetitions_list=[1, 2],
        num_random_sequences=5,
    )
    cb.generate_circuits()

    ideal_sim = qcm.IdealSimulator()
    cb.run(device=ideal_sim, num_shots=1024)
    result = cb.analyze()

    assert "cycle_fidelities" in result
    assert len(result["cycle_fidelities"]) == 2
    assert all(0 <= f <= 1.1 for f in result["cycle_fidelities"])


def test_ptm_elements_in_result(simple_g_layer_1q):
    """Test that PTM elements are stored in result."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_1q,
        repetitions_list=[1, 2],
        num_random_sequences=5,
    )
    cb.generate_circuits()

    ideal_sim = qcm.IdealSimulator()
    cb.run(device=ideal_sim, num_shots=1024)
    result = cb.analyze()

    assert "ptm_elements" in result
    assert 1 in result["ptm_elements"]
    assert 2 in result["ptm_elements"]
    # Each should have PTM values for each Pauli
    assert len(result["ptm_elements"][1]) == 3  # X, Y, Z for 1 qubit


def test_plot_ratio_method(simple_g_layer_1q):
    """Test plotting with ratio method."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_1q,
        repetitions_list=[1, 2, 4],
        num_random_sequences=5,
        fidelity_method="ratio",
    )
    cb.generate_circuits()

    ideal_sim = qcm.IdealSimulator()
    cb.run(device=ideal_sim, num_shots=1024)
    cb.analyze()

    fig, ax = plt.subplots()
    cb._plot(axes=ax)

    assert ax.get_xlabel() == "Number of cycles (m)"
    assert ax.get_ylabel() == "Cycle Fidelity"
    plt.close(fig)


def test_plot_fit_method(simple_g_layer_1q):
    """Test plotting with fit method."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_1q,
        repetitions_list=[1, 2, 4],
        num_random_sequences=5,
        fidelity_method="fit",
    )
    cb.generate_circuits()

    ideal_sim = qcm.IdealSimulator()
    cb.run(device=ideal_sim, num_shots=1024)
    cb.analyze()

    fig, ax = plt.subplots()
    cb._plot(axes=ax)

    assert ax.get_xlabel() == "Number of cycles (m)"
    assert ax.get_ylabel() == "Cycle Fidelity"
    # Check that fitted line exists in legend
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Fit" in text for text in legend_texts)
    plt.close(fig)


def test_has_plotting(simple_g_layer_1q):
    """Test that CycleBenchmarking has plotting functionality."""
    cb = qcm.CycleBenchmarking(
        g_layer=simple_g_layer_1q,
        repetitions_list=[1, 2],
        num_random_sequences=2,
    )
    assert cb.has_plotting() is True


def test_fit_func():
    """Test the exponential fit function."""
    x = np.array([0, 1, 2, 3])
    a, b, c = 1.0, 0.1, 0.0
    result = qcm.CycleBenchmarking.fit_func(x, a, b, c)
    expected = a * np.exp(-b * x) + c
    np.testing.assert_allclose(result, expected)
