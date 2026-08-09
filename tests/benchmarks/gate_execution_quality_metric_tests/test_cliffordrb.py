"""test_cliffordrb.py.

Unit tests for the CliffordRB benchmark in qcmet.benchmarks.cliffordrb.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import qcmet as qcm


def test_raise_error_invalid_gate():
    """Verify that ValueError is raised when a non-Clifford target gate is specified."""
    non_clifford = QuantumCircuit(1)
    non_clifford.t(0)

    with pytest.raises(
        ValueError,
        match="target_clifford is not a valid Clifford gate.",
    ):
        qcm.CliffordRB(
            m_list=[10],
            circs_per_m=1,
            qubits=1,
            target_clifford=non_clifford,
        )


@pytest.mark.parametrize("qubits,identity", [(1, 2), (2, 4)])
def test_circ_operator(qubits, identity):
    """Verify CliffordRB circuits compose to identity after inverse."""
    experiment = qcm.CliffordRB(
        m_list=[0, 20, 40],
        circs_per_m=1,
        qubits=qubits,
        errorbar_method="none",
    )
    experiment.generate_circuits()

    circ = experiment.circuits[-1]
    circ.remove_final_measurements()

    op = qi.Operator(circ)

    assert op == Operator(np.eye(identity))


@pytest.mark.parametrize("qubits,m_max", [(1, 20), (2, 20)])
def test_num_gates(qubits, m_max):
    """Verify correct number of Clifford gates are applied."""
    experiment = qcm.CliffordRB(
        m_list=[0, 20],
        circs_per_m=1,
        qubits=qubits,
        errorbar_method="none",
    )
    experiment.generate_circuits()

    circ = experiment.circuits[-1]
    gates = dict(circ.count_ops())

    gates.pop("barrier", None)
    gates.pop("measure", None)

    total_gates = sum(gates.values())

    assert total_gates == m_max + 1


@pytest.mark.parametrize("qubits", [1, 2])
def test_perfect_emulator(qubits):
    """Verify noiseless CliffordRB returns only the all-zero state."""
    experiment = qcm.CliffordRB(
        m_list=[0, 20, 40],
        circs_per_m=2,
        qubits=qubits,
        errorbar_method="none",
    )
    experiment.generate_circuits()

    ideal_sim = qcm.IdealSimulator()
    experiment.run(device=ideal_sim, num_shots=100)

    ground_state = "0" * qubits

    assert all(
        counts.get(ground_state, 0) == 100
        for counts in experiment.experiment_data["circuit_measurements"]
    )


@pytest.mark.parametrize("qubits", [1, 2])
def test_analyze_ideal(qubits):
    """Verify ideal CliffordRB gives zero average gate error."""
    experiment = qcm.CliffordRB(
        m_list=[0, 20, 40, 60, 80, 100],
        circs_per_m=2,
        qubits=qubits,
        errorbar_method="fit",
    )
    experiment.generate_circuits()

    ideal_sim = qcm.IdealSimulator()
    experiment.run(device=ideal_sim, num_shots=100)

    results = experiment.analyze()

    assert np.isclose(results["AverageGateError"], 0.0, atol=1e-5)
    assert "fit_result" in results
    assert results["fit_result"]["errorbar_method"] == "fit"


@pytest.mark.parametrize("qubits", [1, 2])
def test_analyze_noisy(qubits):
    """Verify noisy CliffordRB gives non-zero average gate error."""
    experiment = qcm.CliffordRB(
        m_list=[0, 20, 50, 100, 200, 500],
        circs_per_m=3,
        qubits=qubits,
        errorbar_method="fit",
    )
    experiment.generate_circuits()

    noisy_sim = qcm.NoisySimulator()
    experiment.run(device=noisy_sim, num_shots=100)

    results = experiment.analyze()

    assert float(results["AverageGateError"]) > 0.0


def test_plot():
    """Verify plot function creates correct axes labels."""
    experiment = qcm.CliffordRB(
        m_list=[0, 20, 40, 60, 80, 100],
        circs_per_m=2,
        qubits=2,
        errorbar_method="fit",
    )
    experiment.generate_circuits()

    noisy_sim = qcm.NoisySimulator()
    experiment.run(device=noisy_sim, num_shots=100)
    experiment.analyze()

    fig, ax = plt.subplots()
    experiment._plot(axes=ax)

    assert ax.get_xlabel() == r"$m$"
    assert ax.get_ylabel() == r"$p_0$"

    plt.close(fig)


def test_bootstrap_errorbars_present():
    """Verify bootstrap errorbar result contains bootstrap samples and uncertainty fields."""
    experiment = qcm.CliffordRB(
        m_list=[0, 20, 40, 80],
        circs_per_m=4,
        qubits=1,
        errorbar_method="bootstrap",
        bootstrap_samples=20,
        bootstrap_seed=123,
        bootstrap_ci_level=0.68,
    )
    experiment.generate_circuits()

    noisy_sim = qcm.NoisySimulator()
    experiment.run(device=noisy_sim, num_shots=100)

    result = experiment.analyze()
    errorbars = result["fit_result"]["errorbars"]

    assert errorbars["method"] == "bootstrap"
    assert errorbars["status"] == "success"
    assert "alpha_stderr" in errorbars
    assert "AverageGateError_stderr" in errorbars
    assert "bootstrap_popt_samples" in errorbars
    assert len(errorbars["bootstrap_popt_samples"]) >= 2


def test_bootstrap_plot_has_confidence_band():
    """Verify bootstrap plotting adds a shaded confidence band."""
    experiment = qcm.CliffordRB(
        m_list=[0, 20, 40, 80],
        circs_per_m=4,
        qubits=1,
        errorbar_method="bootstrap",
        bootstrap_samples=20,
        bootstrap_seed=123,
        bootstrap_ci_level=0.68,
    )
    experiment.generate_circuits()

    noisy_sim = qcm.NoisySimulator()
    experiment.run(device=noisy_sim, num_shots=100)
    experiment.analyze()

    fig, ax = plt.subplots()
    experiment._plot(ax)

    # fill_between adds PolyCollections to ax.collections.
    assert len(ax.collections) >= 1

    plt.close(fig)


@pytest.mark.parametrize("qubits", [1, 2])
def test_with_target_gate(qubits):
    """Verify CliffordRB works with a target Clifford."""
    circ = QuantumCircuit(qubits)
    circ.x(0)

    experiment = qcm.CliffordRB(
        m_list=[10, 50, 100],
        circs_per_m=3,
        qubits=qubits,
        target_clifford=circ,
        errorbar_method="fit",
    )
    experiment.generate_circuits()

    noisy_sim = qcm.NoisySimulator()
    experiment.run(device=noisy_sim, num_shots=100)

    result = experiment.analyze()

    assert result is not None

    fig, ax = experiment.plot()

    assert fig is not None
    assert ax is not None

    plt.close(fig)


def test_result_with_known_error():
    """Verify AverageGateError is comparable to known depolarizing error."""
    all_gate_noise = NoiseModel()
    p_err = 0.01

    error_1q = depolarizing_error(p_err, 1)
    error_2q = depolarizing_error(p_err, 2)

    all_gate_noise.add_all_qubit_quantum_error(
        error_1q,
        ["u1", "u2", "u3"],
        warnings=False,
    )
    all_gate_noise.add_all_qubit_quantum_error(
        error_2q,
        ["cx"],
        warnings=False,
    )

    experiment = qcm.CliffordRB(
        m_list=[0, 1, 2, 3, 5, 10, 15, 20, 30, 50, 70, 100, 200],
        circs_per_m=5,
        qubits=1,
        errorbar_method="fit",
    )
    experiment.generate_circuits()

    noisy_sim = qcm.AerSimulator(
        noise_model=all_gate_noise,
        basis_gates=["u1", "u2", "u3", "cx"],
        seed_simulator=42,
    )
    experiment.run(device=noisy_sim, num_shots=10000)
    experiment.analyze()

    assert np.isclose(
        float(experiment.result["AverageGateError"]),
        p_err * 1 / 2,
        rtol=0.1,
    )