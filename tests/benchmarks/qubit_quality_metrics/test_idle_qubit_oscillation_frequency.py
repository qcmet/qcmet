"""test_idle_qubit_oscillation_frequency.py.

Unit tests for the IdleQubitOscillationFrequency benchmark in qcmet.benchmarks.idle_qubit_oscillation_frequency.
"""

import numpy as np
import pytest
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import qcmet
from qcmet import IdleQubitOscillationFrequency, NoisySimulator
from qcmet.utils import final_statevector


def test_prepare_initial_state():
    """Verify that the initial state is correctly prepared."""
    for extra_zz in [0, 1]:
        idle_qubit_osc_freq = IdleQubitOscillationFrequency(
            dt=1, t_max=1, extra_zz_crosstalk=extra_zz
        )
        for state, ideal_result in zip(
            [0, 1, 2],
            [[1, 0], np.array([1, 1]) / np.sqrt(2), np.array([1, 1j]) / np.sqrt(2)],
            strict=True,
        ):
            if extra_zz == 1:
                ideal_result = np.kron(np.array([1, 1]) / np.sqrt(2), ideal_result)
            test_circuit = QuantumCircuit(extra_zz + 1)
            idle_qubit_osc_freq.prepare_initial_state(test_circuit, state)
            test_circuit.measure_all()
            result = final_statevector(test_circuit)
            assert result.equiv(Statevector(ideal_result))


def test_add_idle_gates():
    """Verify that the idle gates are correctly added."""
    for extra_zz in [0, 1]:
        idle_qubit_osc_freq = IdleQubitOscillationFrequency(
            dt=1, t_max=1, extra_zz_crosstalk=extra_zz
        )
        test_circuit = QuantumCircuit(extra_zz + 1)
        idle_qubit_osc_freq.add_idle_gates(test_circuit, 5)
        assert len(test_circuit.data) == (2 + extra_zz) * 5


def test_change_measurement_basis():
    """Verify that the measurement basis is correctly changed."""
    idle_qubit_osc_freq = IdleQubitOscillationFrequency(dt=1, t_max=1)
    for basis, ideal_result in zip(
        [0, 1, 2],
        [np.array([1, 1]) / np.sqrt(2), [1, 0], np.array([1 - 1j, 1 + 1j]) / 2],
        strict=True,
    ):
        test_circuit = QuantumCircuit(1)
        test_circuit.h(0)
        idle_qubit_osc_freq.change_measurement_basis(test_circuit, basis)
        test_circuit.measure_all()
        result = final_statevector(test_circuit)
        assert result.equiv(Statevector(ideal_result))


def test_generate_circuits():
    """Verify that circuits are generated correctly."""
    for extra_zz in [0, 1]:
        idle_qubit_osc_freq = IdleQubitOscillationFrequency(
            dt=0.5, t_max=15, extra_zz_crosstalk=extra_zz
        )
        circuits = idle_qubit_osc_freq._generate_circuits()
        assert len(circuits) == 15 // 0.5 * 9
        for qc in circuits:
            assert isinstance(qc, QuantumCircuit)
            assert qc.num_qubits == extra_zz + 1


def test_analyze_simulator_without_zz_crosstalk():
    """Verify that without adding ZZ crosstalk to simulate non-Markovian noise, the analysis returns 0."""
    idle_qubit_osc_freq = IdleQubitOscillationFrequency(0.5, 15)

    # Generate circuits
    idle_qubit_osc_freq.generate_circuits()
    # Mock run without shots
    measurements = []
    for c in idle_qubit_osc_freq.experiment_data["circuit"]:
        state = Statevector([1, 0]).evolve(c.remove_final_measurements(False))
        probs = state.probabilities_dict()
        measurements.append(probs)

    idle_qubit_osc_freq.experiment_data["circuit_measurements"] = measurements
    idle_qubit_osc_freq._runtime_params = {"num_shots": 1}
    results = idle_qubit_osc_freq.analyze()
    assert np.allclose(results["idle_qubit_oscillation_frequency"], 0.0, atol=0.001)


def test_analyze_simulator_with_zz_crosstalk():
    """Verify that by adding ZZ crosstalk to simulate non-Markovian noise, the analysis executes correctly."""
    idle_qubit_osc_freq = IdleQubitOscillationFrequency(0.5, 15, 0.15)

    # Generate circuits
    idle_qubit_osc_freq.generate_circuits()

    # Mock run without shots
    measurements = []
    for c in idle_qubit_osc_freq.experiment_data["circuit"]:
        state = Statevector([1, 0, 0, 0]).evolve(c.remove_final_measurements(False))
        probs = qcmet.AerSimulator.reverse_bitstrings(state.probabilities_dict())
        probs_q0 = {"0": 0, "1": 0}
        for k, v in probs.items():
            probs_q0[k[0]] += v
        measurements.append(probs_q0)

    idle_qubit_osc_freq.experiment_data["circuit_measurements"] = measurements
    idle_qubit_osc_freq._runtime_params = {"num_shots": 1}
    results = idle_qubit_osc_freq.analyze()
    assert np.allclose(results["idle_qubit_oscillation_frequency"], 0.3, atol=0.001)


def test_plot():
    """Verify plot function creates a plot with correct axes labels."""
    idle_qubit_osc_freq = IdleQubitOscillationFrequency(0.5, 15, 0.15)
    idle_qubit_osc_freq.generate_circuits()
    device = NoisySimulator()
    idle_qubit_osc_freq.run(device=device, num_shots=1024)
    idle_qubit_osc_freq.analyze()
    fig, ax = plt.subplots()
    idle_qubit_osc_freq._plot(axes=ax)
    assert ax.get_xlabel() == r"t $(n_{\mathrm{1q gates}})$"
    assert ax.get_ylabel() == "Purity"


def test_qubit_index():
    """Test that instructions are given to correct qubit."""
    experiment = IdleQubitOscillationFrequency(0.5, 15, qubit_index=3)
    experiment.generate_circuits()
    circ = experiment.circuits[0]
    assert any(
        ci.operation.name == "measure"
        and 3 in [circ.find_bit(q).index for q in ci.qubits]
        for ci in circ.data
    )
    assert not any(any(circ.find_bit(q).index in [0, 1, 2] for q in ci.qubits)
        for ci in circ.data
    )


def test_crosstalk_qubit_index():
    """Test that instructions are given to correct qubit for crosstalk."""
    experiment = IdleQubitOscillationFrequency(0.5, 15, 0.15, qubit_index=3, crosstalk_qubit_index=5)
    experiment.generate_circuits()
    circ = experiment.circuits[0]
    assert any(
        ci.operation.name == "h"
        and 5 in [circ.find_bit(q).index for q in ci.qubits]
        for ci in circ.data
    )
    assert not any(ci.operation.name == "measure" and any(circ.find_bit(q).index in [0, 1, 2, 4, 5] for q in ci.qubits)
        for ci in circ.data)


def test_qubit_index_not_int():
    """Test that ValueError raised when qubit argument is not a list."""
    with pytest.raises(ValueError, match="Qubits indices must be int"):
        IdleQubitOscillationFrequency(0.5, 15, qubit_index=[2])
