"""Test the Pauli twirling class."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, CZGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager

from qcmet.utils import PauliTwirl


def _two_qubit_op_count(qc: QuantumCircuit) -> int:
    """Count two‑qubit operations using the non‑deprecated API."""
    return sum(1 for instr in qc.data if len(instr.qubits) == 2)


def _count_ops_by_name(qc: QuantumCircuit, name: str) -> int:
    """Count instructions by operation name (e.g., 'cx', 'cz', 'pauli')."""
    return sum(1 for instr in qc.data if instr.operation.name == name)


def test_unitary_equivalence_preserved():
    """Twirl should preserve circuit unitary up to global phase."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    qc.cx(0, 1)
    qc.sx(0)
    qc.cz(0, 1)

    base_u = Operator(qc)

    pm = PassManager([PauliTwirl(seed=7)])
    twirled = pm.run(qc)

    twirled_u = Operator(twirled)
    assert base_u.equiv(twirled_u), "Twirling must preserve unitary up to global phase."


def test_reproducibility_with_seed_qasm3():
    """Setting NumPy's seed should make twirling choices reproducible (compare QASM 3)."""
    qasm3 = pytest.importorskip(
        "qiskit.qasm3", reason="QASM 3 support required for this test."
    )
    dumps = qasm3.dumps

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(0, 1)
    qc.cz(0, 1)

    pm = PassManager([PauliTwirl(seed=1234)])
    twirled_1 = pm.run(qc)

    # Reset the seed and re‑run
    pm = PassManager([PauliTwirl(seed=1234)])
    twirled_2 = pm.run(qc)

    # Exact text equality is appropriate when the RNG is deterministic
    assert dumps(twirled_1) == dumps(twirled_2)


def test_selective_twirling_targets_by_name():
    """Only the gates listed in gates_to_twirl should be twirled (check 'pauli' insertions)."""
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cz(0, 1)

    # Twirl CZ only
    rng = np.random.default_rng(42)
    pm = PassManager([PauliTwirl(gates_to_twirl=[CZGate()], seed=rng)])
    twirled_cz_only = pm.run(qc)

    # Expect 2 Pauli insertions (left & right) around the single CZ
    assert _count_ops_by_name(twirled_cz_only, "cz") == 1
    assert _count_ops_by_name(twirled_cz_only, "cx") == 1
    assert _count_ops_by_name(twirled_cz_only, "pauli") == 2
    # And those 'pauli' ops should be two‑qubit
    assert all(
        len(instr.qubits) == 2
        for instr in twirled_cz_only.data
        if instr.operation.name == "pauli"
    )

    # Twirl CX only
    pm2 = PassManager([PauliTwirl(gates_to_twirl=[CXGate()], seed=42)])
    twirled_cx_only = pm2.run(qc)

    assert _count_ops_by_name(twirled_cx_only, "cx") == 1
    assert _count_ops_by_name(twirled_cx_only, "cz") == 1
    assert _count_ops_by_name(twirled_cx_only, "pauli") == 2
    assert all(
        len(instr.qubits) == 2
        for instr in twirled_cx_only.data
        if instr.operation.name == "pauli"
    )
