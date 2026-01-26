"""Tests for IdealSimulator device."""
from qiskit import QuantumCircuit

import qcmet as qcm


def test_empty_circ():
    """Checks an empty circuit gives the correct result."""
    circ = QuantumCircuit(2)
    circ.measure_all()
    ideal_sim = qcm.IdealSimulator()
    assert ideal_sim.run(circuits=circ, num_shots=1000)["00"] == 1000


def test_x_gate_circ():
    """Check that bitstring ordering is correct."""
    circ = QuantumCircuit(2)
    circ.x(0)
    circ.measure_all()
    ideal_sim = qcm.IdealSimulator()
    assert ideal_sim.run(circuits=circ, num_shots=1000)["10"] == 1000


def test_circs_list():
    """Verify that device can take list of circuits as input."""
    circs = []
    for _ in range(2):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.measure_all()
        circs.append(circ)
    ideal_sim = qcm.IdealSimulator()
    assert ideal_sim.run(circuits=circs, num_shots=1000)[0]["10"] == 1000
