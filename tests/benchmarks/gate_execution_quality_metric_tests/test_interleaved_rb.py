"""test_interleaved_rb.py.

Unit tests for the InterleavedRB benchmark in qcmet.benchmarks.interleaved_rb.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import qcmet as qcm


@pytest.fixture
def x_gate():
    """Fixture to create a single-qubit Clifford target gate."""
    circ = QuantumCircuit(1)
    circ.x(0)
    return circ


@pytest.fixture
def cx_gate():
    """Fixture to create a two-qubit Clifford target gate."""
    circ = QuantumCircuit(2)
    circ.cx(0, 1)
    return circ


def test_raise_error_no_target():
    """Verify TypeError is raised when no target_clifford is specified."""
    with pytest.raises(TypeError):
        qcm.InterleavedRB(m_list=[10], circs_per_m=1, qubits=1)


@pytest.mark.parametrize(
    "qubits, identity, target_gate",
    [(1, 2, "x_gate"), (2, 4, "cx_gate")],
)
def test_circ_operator(qubits, identity, target_gate, request):
    """Verify IRB circuits compose to identity after inverse."""
    experiment = qcm.InterleavedRB(
        m_list=[10],
        circs_per_m=1,
        qubits=qubits,
        target_clifford=request.getfixturevalue(target_gate),
        errorbar_method="none",
    )
    experiment.generate_circuits()

    interleaved_circ = experiment.experiment_data[
        experiment.experiment_data["type"] == "IRB"
    ].iloc[0]["circuit"]

    interleaved_circ.remove_final_measurements()

    op = qi.Operator(interleaved_circ)

    assert op == Operator(np.eye(identity))


@pytest.mark.parametrize("seq_length", [10, 50, 100])
def test_num_target_gates(seq_length, x_gate):
    """Verify target Clifford appears at least seq_length times in IRB circuit."""
    experiment = qcm.InterleavedRB(
        m_list=[seq_length],
        circs_per_m=1,
        qubits=1,
        target_clifford=x_gate,
        errorbar_method="none",
    )
    experiment.generate_circuits()

    interleaved_circ = experiment.experiment_data[
        experiment.experiment_data["type"] == "IRB"
    ].iloc[0]["circuit"]

    interleaved_circ.remove_final_measurements()

    gates = dict(interleaved_circ.count_ops())
    gates.pop("barrier", None)
    gates.pop("unitary", None)

    assert gates["x"] >= seq_length


@pytest.mark.parametrize("qubits, target_gate", [(1, "x_gate"), (2, "cx_gate")])
def test_analyze_noiseless(qubits, target_gate, request):
    """Verify ideal IRB gives zero average and interleaved gate errors."""
    experiment = qcm.InterleavedRB(
        m_list=[0, 20, 50, 100, 300],
        circs_per_m=5,
        qubits=qubits,
        target_clifford=request.getfixturevalue(target_gate),
        errorbar_method="fit",
    )
    experiment.generate_circuits()

    ideal_sim = qcm.IdealSimulator()
    experiment.run(device=ideal_sim, num_shots=100)

    results = experiment.analyze()

    assert np.isclose(results["AverageGateError"], 0, atol=1e-5)
    assert np.isclose(results["InterleavedGateError"], 0, atol=1e-5)
    assert "fit_result" in results
    assert results["fit_result"]["final_fit_method"] == "joint_curve_fit"


@pytest.mark.parametrize("qubits, target_gate", [(1, "x_gate"), (2, "cx_gate")])
def test_analyze_noisy_joint_fit(qubits, target_gate, request):
    """Verify joint IRB fitted curve usually decays faster than RB curve."""
    experiment = qcm.InterleavedRB(
        m_list=[0, 20, 50, 100, 300],
        circs_per_m=5,
        qubits=qubits,
        target_clifford=request.getfixturevalue(target_gate),
        errorbar_method="fit",
    )
    experiment.generate_circuits()

    noisy_sim = qcm.NoisySimulator()
    experiment.run(device=noisy_sim, num_shots=100)
    experiment.analyze()

    fit_xxs = np.linspace(0, max(experiment.config["m_list"]) + 1, 1000)

    rb_is_irb = np.zeros_like(fit_xxs, dtype=bool)
    irb_is_irb = np.ones_like(fit_xxs, dtype=bool)

    popt = experiment.fit_result["fit_result"]["popt"]

    fit_rb = experiment._fit_func((fit_xxs, rb_is_irb), *popt)
    fit_irb = experiment._fit_func((fit_xxs, irb_is_irb), *popt)

    condition = fit_irb < fit_rb

    assert np.mean(condition) > 0.9


def test_interleaved_bootstrap_errorbars_present(x_gate):
    """Verify joint IRB bootstrap errorbars contain samples and key uncertainty fields."""
    experiment = qcm.InterleavedRB(
        m_list=[0, 20, 40, 80],
        circs_per_m=4,
        qubits=1,
        target_clifford=x_gate,
        errorbar_method="bootstrap",
        bootstrap_samples=20,
        bootstrap_seed=123,
        bootstrap_confidence_level=0.68,
    )
    experiment.generate_circuits()

    noisy_sim = qcm.NoisySimulator()
    experiment.run(device=noisy_sim, num_shots=100)

    result = experiment.analyze()
    errorbars = result["fit_result"]["errorbars"]

    assert errorbars["method"] == "bootstrap"
    assert errorbars["status"] == "success"
    assert "alpha_stderr" in errorbars
    assert "alpha_g_stderr" in errorbars
    assert "InterleavedGateError_stderr" in errorbars
    assert "bootstrap_popt_samples" in errorbars
    assert len(errorbars["bootstrap_popt_samples"]) >= 2


def test_interleaved_bootstrap_plot_has_confidence_bands(x_gate):
    """Verify joint IRB bootstrap plotting adds shaded RB and IRB confidence bands."""
    experiment = qcm.InterleavedRB(
        m_list=[0, 20, 40, 80],
        circs_per_m=4,
        qubits=1,
        target_clifford=x_gate,
        errorbar_method="bootstrap",
        bootstrap_samples=20,
        bootstrap_seed=123,
        bootstrap_confidence_level=0.68,
    )
    experiment.generate_circuits()

    noisy_sim = qcm.NoisySimulator()
    experiment.run(device=noisy_sim, num_shots=100)
    experiment.analyze()

    fig, ax = plt.subplots()
    experiment._plot(ax)

    # Two shaded confidence bands should be present: one for RB and one for IRB.
    assert len(ax.collections) >= 2

    plt.close(fig)


def test_result_with_known_error_x_gate():
    """Verify InterleavedGateError is comparable to known x-gate depolarizing error."""
    x_gate_noise = NoiseModel()
    p_err = 0.01

    error_1q = depolarizing_error(p_err, 1)
    x_gate_noise.add_all_qubit_quantum_error(error_1q, ["x"], warnings=False)

    circ = QuantumCircuit(1)
    circ.x(0)

    experiment = qcm.InterleavedRB(
        m_list=[0, 1, 2, 3, 5, 10, 15, 20, 30, 50, 70, 100, 200],
        circs_per_m=5,
        qubits=1,
        target_clifford=circ,
        errorbar_method="fit",
    )
    experiment.generate_circuits()

    noisy_sim = qcm.AerSimulator(
        noise_model=x_gate_noise,
        seed_simulator=42,
    )
    experiment.run(device=noisy_sim, num_shots=10000)
    experiment.analyze()

    assert np.isclose(
        float(experiment.result["InterleavedGateError"]),
        p_err * 1 / 2,
        rtol=0.1,
    )


def test_result_with_known_error_cx_gate():
    """Verify InterleavedGateError is comparable to known cx-gate depolarizing error."""
    cx_gate_noise = NoiseModel()
    p_err = 0.01

    error_2q = depolarizing_error(p_err, 2)
    cx_gate_noise.add_all_qubit_quantum_error(error_2q, ["cx"], warnings=False)

    circ = QuantumCircuit(2)
    circ.cx(0, 1)

    experiment = qcm.InterleavedRB(
        m_list=[0, 1, 2, 3, 5, 10, 15, 20, 30, 50, 70, 100, 150, 200],
        circs_per_m=10,
        qubits=2,
        target_clifford=circ,
        errorbar_method="fit",
    )
    experiment.generate_circuits()

    noisy_sim = qcm.AerSimulator(
        noise_model=cx_gate_noise,
        seed_simulator=42,
    )
    experiment.run(device=noisy_sim, num_shots=10000)
    experiment.analyze()

    assert np.isclose(
        float(experiment.result["InterleavedGateError"]),
        p_err * 3 / 4,
        rtol=0.1,
    )
