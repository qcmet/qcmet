"""Tests for over- or under-rotation angle."""

import numpy as np
import pytest
from qiskit.circuit.library import HGate

from qcmet import NoisySimulator, OverUnderRotationAngle


def test_check_num_gates_for_id_accepts_default():
    """Accepts 4 x SX as a valid pseudoidentity."""
    OverUnderRotationAngle(qubits=1, delta_m=1, m_max=1)


def test_check_num_gates_for_id_rejects_bad_count():
    """Rejects invalid pseudoidentity repeat counts."""
    # 2 × SX is not identity, should raise
    with pytest.raises(ValueError):
        OverUnderRotationAngle(qubits=1, delta_m=1, m_max=1, num_gates_for_id=2)


def test_check_num_gates_for_id_with_different_gate():
    """Accepts H^2 = I as a valid pseudoidentity."""
    OverUnderRotationAngle(qubits=1, delta_m=1, m_max=1, gate=HGate, num_gates_for_id=2)


def test_generate_circuits_metadata_and_length():
    """Generates circuits with correct 'm', 'id', and 'hash' fields."""
    bench = OverUnderRotationAngle(qubits=1, delta_m=1, m_max=2)
    records = bench._generate_circuits()
    m_array = bench.config["m_array"]

    assert isinstance(records, list)
    assert [rec["m"] for rec in records] == list(m_array)

    for rec in records:
        assert "circuit" in rec and hasattr(rec["circuit"], "draw")
        assert "id" in rec and isinstance(rec["id"], str)
        assert "hash" in rec and isinstance(rec["hash"], str)


def test_fit_func_simple():
    """fit_func matches manual formula for given parameters."""
    a, b, decay, theta, phase = 0.1, 0.5, 0.2, 0.3, 0.4
    m = np.array([0, 1, 2, 3])
    y = OverUnderRotationAngle.fit_func(m, a, b, decay, theta, phase)
    expected = b * np.exp(-decay * m) * np.cos(theta * m + phase) + a
    assert np.allclose(y, expected)


def test_full_pipeline_on_noisy_simulator(tmp_path):
    """Full generate,run,analyze returns expected rotation on noisy simulator."""
    bench = OverUnderRotationAngle(
        qubits=1, delta_m=2, m_max=100, save_path=tmp_path
    )
    bench.generate_circuits()
    overrotation_amount= np.pi/100
    sim = NoisySimulator(t1=0, t2=0, overrotation_amount=overrotation_amount)
    shots = 512
    bench.run(sim, num_shots = shots)
    result = bench.analyze()
    assert result["success"]
    assert np.isclose(result["OverUnderRotationAngle"], overrotation_amount, atol=1e-2)

