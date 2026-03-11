"""test_interleaved_rb.py.

Unit tests for the InterleavedRB benchmark in qcmet.benchmarks.interleaved_rb.
"""

import numpy as np
import pytest
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.operators import Operator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
)

import qcmet as qcm


def test_raise_error_no_target():
    """Verify that the ValueError is raised when no target_clifford is specified."""
    with pytest.raises(TypeError):
        qcm.InterleavedRB(m_list=[10], circs_per_m=1, qubits=1)


@pytest.mark.parametrize("qubits,identity", [(1, 2), (2, 4)])
def test_circ_operator(qubits, identity):
    """Verify that the total operations acting on InterleavedRB circuits are equal to the identity operator."""
    circ = QuantumCircuit(qubits)
    circ.x(0)
    experiment = qcm.InterleavedRB(
        m_list=[10], circs_per_m=1, qubits=qubits, target_clifford=circ
    )
    experiment.generate_circuits()
    interleaved_circ = experiment.experiment_data[
        experiment.experiment_data["type"] == "IRB"
    ].loc[1, "circuit"]
    interleaved_circ.remove_final_measurements()
    op = qi.Operator(interleaved_circ)
    assert (
        op == Operator(np.eye(identity))
    )  # asserts circuit operator == identity matrix operator (of dimensions: identity x identity)


@pytest.mark.parametrize("seq_length", [10, 50, 100])
def test_num_target_gates(seq_length):
    """Verify that the number of target clifford gates is at least 50% of the total clifford gates."""
    circ = QuantumCircuit(1)
    circ.x(0)
    experiment = qcm.InterleavedRB(
        m_list=[seq_length], circs_per_m=1, qubits=1, target_clifford=circ
    )
    experiment.generate_circuits()
    interleaved_circ = experiment.experiment_data[
        experiment.experiment_data["type"] == "IRB"
    ].loc[1, "circuit"]
    interleaved_circ.remove_final_measurements()
    gates = dict(interleaved_circ.count_ops())
    gates.pop("barrier")
    gates.pop("unitary")
    assert gates["x"] >= seq_length


@pytest.mark.parametrize("qubits", [1, 2])
def test_analyze_noiseless(qubits):
    """Verify that analyze() computes average and interleaved gate errors to be zero when running on noiseless device."""
    circ = QuantumCircuit(qubits)
    circ.x(0)
    experiment = qcm.InterleavedRB(
        m_list=[0, 20, 50, 100, 300, 500, 1000],
        circs_per_m=5,
        qubits=qubits,
        target_clifford=circ,
    )
    experiment.generate_circuits()
    ideal_sim = qcm.IdealSimulator()
    experiment.run(device=ideal_sim, num_shots=100)
    results = experiment.analyze()
    assert np.isclose(results["AverageGateError"], 0, atol=1e-5)
    assert np.isclose(results["InterleavedGateError"], 0, atol=1e-5)


@pytest.mark.parametrize("qubits", [1, 2])
def test_analyze_noisy(qubits):
    """Verify that at a given m, IRB p_surv < RB p_surv for at least 90% of all data points."""
    circ = QuantumCircuit(qubits)
    circ.x(0)
    experiment = qcm.InterleavedRB(
        m_list=[0, 20, 50, 100, 300, 500, 1000],
        circs_per_m=5,
        qubits=qubits,
        target_clifford=circ,
    )
    experiment.generate_circuits()
    noisy_sim = qcm.NoisySimulator()
    experiment.run(device=noisy_sim, num_shots=100)
    experiment.analyze()

    fit_xxs = np.linspace(0, (max(experiment.config["m_list"])) + 1, 1000)
    fit_irb = qcm.CliffordRB.fit_func(
        fit_xxs, *experiment.irb_experiment.fit_result["fit_result"]["popt"]
    )
    fit_rb = qcm.CliffordRB.fit_func(
        fit_xxs, *experiment.rb_experiment.fit_result["fit_result"]["popt"]
    )

    condition = [
        irb_fit_datapoint < rb_fit_datapoint
        for irb_fit_datapoint, rb_fit_datapoint in zip(fit_irb, fit_rb, strict=True)
    ]
    assert np.mean(condition) > 0.9


def test_result_with_known_error():
    """Verify that analyze() returns AverageGateError value comparable to noise model."""
    all_gate_noise = NoiseModel()
    p_err = 0.01
    error_1q = depolarizing_error(p_err, 1)
    all_gate_noise.add_all_qubit_quantum_error(error_1q, ["x"], warnings=False)

    q_reg = QuantumRegister(1, name="q")
    circ = QuantumCircuit(q_reg)
    circ.x(0)

    experiment = qcm.InterleavedRB(
        m_list=[0, 1, 2, 3, 5, 10, 15, 20, 30, 50, 70, 100, 200, 300, 400, 500],
        circs_per_m=5,
        qubits=1,
        target_clifford=circ,
    )
    experiment.generate_circuits()
    noisy_sim = qcm.AerSimulator(noise_model=all_gate_noise, seed_simulator=42)
    experiment.run(device=noisy_sim, num_shots=10000)
    experiment.analyze()
    assert np.isclose(
        float(experiment.result["InterleavedGateError"]), p_err * 1 / 2, rtol=0.1
    )
