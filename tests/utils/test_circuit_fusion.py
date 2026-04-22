"""test_circuit_fusion.py.

Unit tests for qcmet.utils.circuit_fusion.
"""

import pytest
from qiskit import QuantumCircuit

from qcmet.utils.circuit_fusion import fuse_circuit_groups


def _single_qubit_measured_circuit(active_qubit, apply_x=True):
    """Build a one-qubit measured circuit on a chosen global qubit index."""
    circuit = QuantumCircuit(active_qubit + 1, 1)
    if apply_x:
        circuit.x(active_qubit)
    circuit.measure(active_qubit, 0)
    return circuit


def _two_qubit_measured_circuit(active_qubits, apply_x_to: list[int]):
    """Build a two-qubit measured circuit on chosen global qubit indices."""
    circuit = QuantumCircuit(max(active_qubits) + 1, 2)
    for idx, qubit in enumerate(active_qubits):
        if idx in apply_x_to:
            circuit.x(qubit)
        circuit.measure(qubit, idx)
    return circuit


def test_fuse_circuit_groups_strict_mode():
    """Verify that strict fusion returns one fused circuit per circuit index."""
    group0 = [
        _single_qubit_measured_circuit(0, apply_x=True),
        _single_qubit_measured_circuit(0, apply_x=False),
    ]
    group1 = [
        _two_qubit_measured_circuit([1, 2], apply_x_to=[0, 1]),
        _two_qubit_measured_circuit([1, 2], apply_x_to=[1]),
    ]

    fused_circuits, clbit_layout = fuse_circuit_groups([group0, group1], fuse_mode="strict")

    assert len(fused_circuits) == 2
    assert clbit_layout == [[0], [1, 2]]
    assert fused_circuits[0].num_qubits == 3
    assert fused_circuits[0].num_clbits == 3


def test_fuse_circuit_groups_min_mode():
    """Verify that min fusion truncates to the smallest group length."""
    group0 = [_single_qubit_measured_circuit(0)]
    group1 = [
        _single_qubit_measured_circuit(1),
        _single_qubit_measured_circuit(1, apply_x=False),
    ]

    fused_circuits, _ = fuse_circuit_groups([group0, group1], fuse_mode="min")

    assert len(fused_circuits) == 1


def test_fuse_circuit_groups_pad_mode():
    """Verify that pad fusion extends to the longest group length."""
    group0 = [_single_qubit_measured_circuit(0)]
    group1 = [
        _single_qubit_measured_circuit(1),
        _single_qubit_measured_circuit(1, apply_x=False),
    ]

    fused_circuits, clbit_layout = fuse_circuit_groups([group0, group1], fuse_mode="pad")

    assert len(fused_circuits) == 2
    assert clbit_layout == [[0], [1]]
    assert fused_circuits[1].num_clbits == 2


def test_fuse_circuit_groups_invalid_mode_raises_error():
    """Verify that an invalid fuse mode raises an error."""
    with pytest.raises(ValueError):
        fuse_circuit_groups([[_single_qubit_measured_circuit(0)]], fuse_mode="bad")


def test_fuse_circuit_groups_empty_group_raises_error():
    """Verify that empty circuit groups are rejected."""
    with pytest.raises(ValueError):
        fuse_circuit_groups([[]])


def test_fuse_circuit_groups_non_circuit_entry_raises_error():
    """Verify that non-circuit entries are rejected."""
    with pytest.raises(ValueError):
        fuse_circuit_groups([["not a circuit"]])


def test_fuse_circuit_groups_inconsistent_clbits_raises_error():
    """Verify that inconsistent clbit counts within a group are rejected."""
    circuit0 = QuantumCircuit(1, 1)
    circuit0.measure(0, 0)

    circuit1 = QuantumCircuit(1, 2)
    circuit1.measure(0, 0)

    with pytest.raises(ValueError):
        fuse_circuit_groups([[circuit0, circuit1]])


def test_fuse_circuit_groups_overlapping_qubits_raises_error():
    """Verify that overlapping active qubits across groups are rejected."""
    group0 = [_single_qubit_measured_circuit(0)]
    group1 = [_single_qubit_measured_circuit(0)]

    with pytest.raises(ValueError):
        fuse_circuit_groups([group0, group1])


def test_fuse_circuit_groups_merges_barriers():
    """Verify that aligned barriers from different groups are merged."""
    circuit0 = QuantumCircuit(1, 1)
    circuit0.x(0)
    circuit0.barrier(0)
    circuit0.z(0)
    circuit0.measure(0, 0)

    circuit1 = QuantumCircuit(2, 1)
    circuit1.x(1)
    circuit1.y(1)
    circuit1.barrier(1)
    circuit1.z(1)
    circuit1.measure(1, 0)

    fused_circuits, _ = fuse_circuit_groups([[circuit0], [circuit1]])
    fused = fused_circuits[0]

    barriers = [inst for inst in fused.data if inst.operation.name == "barrier"]
    assert len(barriers) == 1
    barrier_qubits = {fused.find_bit(q).index for q in barriers[0].qubits}
    assert barrier_qubits == {0, 1}
