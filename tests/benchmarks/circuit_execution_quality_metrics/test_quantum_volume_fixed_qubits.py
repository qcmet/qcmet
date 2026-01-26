"""test_quantum_volume.py.

Unit tests for the QuantumVolumeFixedQubits benchmark in
qcmet.benchmarks.quantum_volume_fixed_qubits.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest import approx
from qiskit import QuantumCircuit

import qcmet as qcm
from qcmet.benchmarks import QuantumVolumeFixedQubits
from qcmet.utils import compute_ideal_outputs, final_statevector


@pytest.mark.parametrize("n", [(1), (2), (3)])
def test_random_complex_matrix_shape(n):
    """Verify that _random_complex_matrix has correct shape."""
    qv = QuantumVolumeFixedQubits()
    matrix = qv._random_complex_matrix(n=n)
    assert matrix.shape == (n, n)


@pytest.mark.parametrize("n", [(1), (2), (3)])
def test_random_complex_matrix_complex(n):
    """Verify that each element in _random_complex_matrix is complex."""
    qv = QuantumVolumeFixedQubits()
    matrix = qv._random_complex_matrix(n=n)
    for element in matrix.flat:
        assert isinstance(element, complex)


def test_haar_measure_determinant():
    """Verify that the matrix generated from the haar measure is unitary."""
    qv = QuantumVolumeFixedQubits()
    haar = qv._haar_measure(4)
    determinant = np.linalg.det(haar)
    assert np.allclose(abs(determinant), 1)


def test_swap_layer():
    """Verify that there is an instance of a swap gate in a circuit when the swap layer is applied (in large qubit limit)."""
    qc = QuantumCircuit(30)
    qv = QuantumVolumeFixedQubits()
    qv._apply_swap_layer(qc, 30)
    assert any(instr.name == "swap" for instr in qc.data)


@pytest.mark.parametrize("qubits,su4", [(2, 2), (3, 3), (4, 8), (5, 10), (6, 18)])
def test_generate_circuits_su4_layers(qubits, su4):
    """Verify that the correct number of su4 gates are applied to the circuit."""
    experiment = qcm.QuantumVolumeFixedQubits(qubits=qubits, trials=1)
    experiment.generate_circuits()
    circ = experiment.circuits[0]
    su4_gates = dict(circ.count_ops())["unitary"]
    assert su4_gates == su4


@pytest.mark.parametrize("qubits", [(1), (2), (3)])
def test_generate_circuits_trials(qubits):
    """Verify that the correct number of circuits are generated."""
    trials = 10
    experiment = qcm.QuantumVolumeFixedQubits(qubits=qubits, trials=trials)
    experiment.generate_circuits()
    assert len(experiment.circuits) == trials


def test_get_final_sv():
    """Verify that the correct statevector is returned for a 2-qubit bell state."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    sv = final_statevector(qc)
    format_sv = np.array([x.real if x.imag == 0 else x for x in sv])
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert np.allclose(expected, format_sv)


def test_get_ideal_outputs():
    """Verify that the correct ideal outputs are returned for a 2-qubit bell state."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    outputs = compute_ideal_outputs(qc)
    expected_outputs = {"00": 0.5, "01": 0, "10": 0, "11": 0.5}
    for key in outputs:
        assert outputs[key] == approx(expected_outputs[key], rel=1e-6)


def test_get_ideal_outputs_ordering():
    """Verify that the ordering of the bitstrings are correct."""
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.measure_all()
    outputs = compute_ideal_outputs(qc)
    expected_output = {"10": 1.0, "00": 0.0, "01": 0.0, "11": 0.0}
    for key in outputs:
        assert outputs[key] == approx(expected_output[key], rel=1e-6)


def test_heavy_outputs():
    """Verify that the correct heavy outputs are returned for a 2-qubit bell state."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    qv = QuantumVolumeFixedQubits()
    ideal_outputs = compute_ideal_outputs(
        qc
    )  # ideal_heavy_outputs requires self.ideal_outputs
    ideal_heavy_outputs = ["00", "11"]
    assert ideal_heavy_outputs == qv._get_heavy_outputs(ideal_outputs=ideal_outputs)


def test_heavy_output_counts():
    """Verify that the heavy output counts are equal to the noiseless simulator counts for a 2-qubit bell state."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    ideal_sim = qcm.IdealSimulator()
    counts = ideal_sim.run(circuits=qc, num_shots=1000)
    qv = QuantumVolumeFixedQubits()
    ideal_outputs = compute_ideal_outputs(qc)
    qv._get_heavy_outputs(ideal_outputs=ideal_outputs)
    p_h = qv._get_heavy_output_counts(counts=counts)
    assert sum(p_h) == 1000


@pytest.mark.parametrize("qubits", [(1), (2), (3)])
def test_analyze_correct_outcome(qubits):
    """Verify that the success outcome returns the correct result."""
    experiment = QuantumVolumeFixedQubits(trials=100, qubits=qubits)
    experiment.generate_circuits()
    ideal_sim = qcm.IdealSimulator()
    experiment.run(device=ideal_sim, num_shots=1000)
    if experiment.analyze()["mean-2sigma"] > 2 / 3:
        assert experiment.analyze()["outcome"] == "Pass"
    else:
        assert experiment.analyze()["outcome"] == "Fail"


@pytest.mark.parametrize("qubits", [(2), (3), (4)])
def test_ideal_analyze(qubits):
    """Verify that a noisless device passes the success criterion."""
    experiment = QuantumVolumeFixedQubits(trials=100, qubits=qubits)
    experiment.generate_circuits()
    ideal_sim = qcm.IdealSimulator()
    experiment.run(device=ideal_sim, num_shots=1000)
    assert experiment.analyze()["outcome"] == "Pass"


def test_plot():
    """Verify plot function creates a plot with correct axes labels."""
    experiment = qcm.QuantumVolumeFixedQubits(qubits=2, trials=100)
    experiment.generate_circuits()
    ideal_sim = qcm.IdealSimulator()
    experiment.run(device=ideal_sim, num_shots=1024)
    experiment.analyze()
    fig, ax = plt.subplots()
    experiment._plot(axes=ax)
    assert ax.get_xlabel() == "Occurences"
    assert ax.get_ylabel() == r"$p_h$"


def test_seed_circuits():
    """Verify that experiments with the same seed produce the same circuits."""
    seed = seed = np.random.randint(1, 100)
    experiment1 = qcm.QuantumVolumeFixedQubits(qubits=3, trials=10, seed=seed)
    experiment1.generate_circuits()
    experiment2 = qcm.QuantumVolumeFixedQubits(qubits=3, trials=10, seed=seed)
    experiment2.generate_circuits()

    for _, (circ1, circ2) in enumerate(
        zip(experiment1.circuits, experiment2.circuits, strict=False)
    ):
        assert circ1 == circ2


def test_seed_result():
    """Verify that experiments with the same seed produce the same result outcome."""
    seed = seed = np.random.randint(1, 100)
    experiment1 = qcm.QuantumVolumeFixedQubits(qubits=2, trials=10, seed=seed)
    experiment1.generate_circuits()
    ideal_sim = qcm.IdealSimulator()
    experiment1.run(device=ideal_sim, num_shots=1024)
    results1 = experiment1.analyze()
    experiment2 = qcm.QuantumVolumeFixedQubits(qubits=2, trials=10, seed=seed)
    experiment2.generate_circuits()
    experiment2.run(device=ideal_sim, num_shots=1024)
    results2 = experiment2.analyze()
    assert results1["outcome"] == results2["outcome"]
