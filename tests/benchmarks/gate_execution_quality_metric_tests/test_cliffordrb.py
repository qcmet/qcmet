"""test_cliffordrb.py.

Unit tests for the CliffordRB benchmark in qcmet.benchmarks.cliffordrb.
"""

import matplotlib.pyplot as plt
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


def test_raise_error_invalid_gate():
    """Verify that the ValueError is raised when a non-Clifford gate is specified."""
    non_clifford = QuantumCircuit(1)
    non_clifford.t(0)
    with pytest.raises(
        ValueError, match="target_clifford is not a valid Clifford gate."
    ):
        qcm.CliffordRB(
            m_list=[10], circs_per_m=1, qubits=1, target_clifford=non_clifford
        )


@pytest.mark.parametrize("qubits,identity", [(1, 2), (2, 4)])
def test_circ_operator(qubits, identity):
    """Verify that the total operations acting on CliffordRB circuits are equal to the identity operator."""
    experiment = qcm.CliffordRB(m_list=[0, 20, 40], circs_per_m=1, qubits=qubits)
    experiment.generate_circuits()
    circs = experiment.circuits
    circ = circs[-1]
    circ.remove_final_measurements()
    op = qi.Operator(circ)
    assert (
        op == Operator(np.eye(identity))
    )  # asserts circuit operator == identity matrix operator (of dimensions: identity x identity)


@pytest.mark.parametrize("qubits,m_max", [(1, 20), (2, 20)])
def test_num_gates(qubits, m_max):
    """Verify that the correct number of clifford gates are applied to the CliffordRB circuit."""
    experiment = qcm.CliffordRB(m_list=[0, 20], circs_per_m=1, qubits=qubits)
    experiment.generate_circuits()
    circs = experiment.circuits
    circ = circs[-1]
    gates = dict(circ.count_ops())
    gates.pop("barrier")
    gates.pop("measure")
    total_gates = sum(gates.values())
    assert total_gates == m_max + 1


@pytest.mark.parametrize("qubits,ground_state_0s", [(1, 1), (2, 2)])
def test_perfect_emulator(qubits, ground_state_0s):
    """Verify that all measurements of the CliffordRB circuit are in the ground state when running the circuit on a noisless device."""
    experiment = qcm.CliffordRB(
        m_list=[0, 20, 40, 60, 80, 100], circs_per_m=2, qubits=qubits
    )
    experiment.generate_circuits()
    ideal_sim = qcm.IdealSimulator()
    experiment.run(device=ideal_sim, num_shots=100)
    assert all(
        1024
        for ground_counts in experiment.experiment_data["circuit_measurements"].apply(
            lambda x: x["0" * ground_state_0s]
        )
    )


@pytest.mark.parametrize("qubits,average_gate_error", [(1, 0), (2, 0)])
def test_analyze_ideal(qubits, average_gate_error):
    """Verify that analyze() computes an average gate error of 0 when running the circuit on a noiseless device."""
    experiment = qcm.CliffordRB(
        m_list=[0, 20, 40, 60, 80, 100], circs_per_m=2, qubits=qubits
    )
    experiment.generate_circuits()
    ideal_sim = qcm.IdealSimulator()
    experiment.run(device=ideal_sim, num_shots=100)
    results = experiment.analyze()
    assert np.isclose(results["AverageGateError"], average_gate_error, atol=1e-5)


@pytest.mark.parametrize("qubits,zero_gate_error", [(1, 0), (2, 0)])
def test_analyze_noisy(qubits, zero_gate_error):
    """Verify that analyze() computes an non-zero average gate error when running the circuit on a noisy device."""
    experiment = qcm.CliffordRB(
        m_list=[0, 20, 50, 100, 200, 4000, 1000], circs_per_m=2, qubits=qubits
    )
    experiment.generate_circuits()
    dummy_sim = qcm.NoisySimulator()
    experiment.run(device=dummy_sim, num_shots=100)
    results = experiment.analyze()
    assert float(results["AverageGateError"]) > zero_gate_error


def test_plot():
    """Verify plot function creates a plot with correct axes labels."""
    experiment = qcm.CliffordRB(
        m_list=[0, 20, 40, 60, 80, 100], circs_per_m=2, qubits=2
    )
    experiment.generate_circuits()
    dummy_sim = qcm.NoisySimulator()
    experiment.run(device=dummy_sim, num_shots=100)
    experiment.analyze()
    fig, ax = plt.subplots()
    experiment._plot(axes=ax)
    assert ax.get_xlabel() == r"$m$"
    assert ax.get_ylabel() == r"$p_0$"


@pytest.mark.parametrize("qubits", [(1), (2)])
def test_with_target_gate(qubits):
    """Verify that CliffordRB works with a target Clifford."""
    q_reg = QuantumRegister(qubits, name="q")
    circ = QuantumCircuit(q_reg)
    circ.x(0)
    experiment = qcm.CliffordRB(
        m_list=[10, 50, 100, 150, 200, 400, 600],
        circs_per_m=3,
        qubits=qubits,
        target_clifford=circ,
    )
    experiment.generate_circuits()
    noisy_sim = qcm.NoisySimulator()
    experiment.run(device=noisy_sim, num_shots=100)
    result = experiment.analyze()
    assert result is not None
    fig, ax = experiment.plot()
    assert fig is not None and ax is not None


def test_result_with_known_error():
    """Verify that analyze() returns AverageGateError value comparable to noise model."""
    all_gate_noise = NoiseModel()
    p_err = 0.01
    error_1q = depolarizing_error(p_err, 1)
    error_2q = depolarizing_error(p_err, 2)
    all_gate_noise.add_all_qubit_quantum_error(
        error_1q, ["u1", "u2", "u3"], warnings=False
    )
    all_gate_noise.add_all_qubit_quantum_error(error_2q, ["cx"], warnings=False)

    experiment = qcm.CliffordRB(
        m_list=[0, 1, 2, 3, 5, 10, 15, 20, 30, 50, 70, 100, 200, 300, 400, 500],
        circs_per_m=5,
        qubits=1,
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
        float(experiment.result["AverageGateError"]), p_err * 1 / 2, rtol=0.1
    )
