"""test_sequential_benchmark.py.

Unit tests for the SequentialBenchmark in qcmet.benchmarks.sequential_benchmark.
"""
import pytest
from qiskit import QuantumCircuit

from qcmet.benchmarks import BaseBenchmark, SequentialBenchmark
from qcmet.devices import IdealSimulator


class DummyBenchmark(BaseBenchmark):
    """Concrete BaseBenchmark for testing SequentialBenchmark methods."""

    def __init__(self, qubits, parameter, **kwargs):
        """Initialize dummy with a parameter."""
        super().__init__("Dummy", qubits, **kwargs)
        self.parameter = parameter

    def _generate_circuits(self):
        c = QuantumCircuit(max(self.qubits) + 1)
        for i in range(max(self.qubits) + 1):
            c.x(i)
        c.measure_all()
        return [c]

    def _analyze(self):
        return {"max_qubit": max(self.qubits) + 1, "parameter": self.parameter}

class DummySequentialBenchmark(SequentialBenchmark):
    """Concrete SequentialBenchmark for testing SequentialBenchmark methods."""

    def __init__(self):
        """Initialize dummy sequential benchmark to run from 2 qubit to 4 qubits."""
        super().__init__(
            "DummySequentialBenchmark",
            DummyBenchmark,
            2,
            4,
            fixed_parameters={"parameter": "some value"}
        )
        self.fail_number = 3

    def should_stop(self, results):
        """Check stopping condition."""
        return results["max_qubit"] >= self.fail_number

    def set_stopping_num(self, fail_number):
        """Change the fail condition for testing purposes."""
        self.fail_number = fail_number

    def _analyze(self):
        return {"analysis": "some text"}

@pytest.fixture
def dummy_sb_instance():
    """Fixture to create a dummy benchmark instance with qubit number from 2 to 4."""
    dummy = DummySequentialBenchmark()
    return dummy


def test_get_current_qubits(dummy_sb_instance):
    """Verify that Sequential correctly gets the number of qubits or qubit indices."""
    qubits = dummy_sb_instance._get_current_qubits()
    assert qubits == 2
    dummy_sb_instance.run_index = 1
    qubits = dummy_sb_instance._get_current_qubits()
    assert qubits == 3

    dummy_sb_instance.run_index = 0
    dummy_sb_instance.config["qubit_indices"] = [33, 22, 11]
    qubits = dummy_sb_instance._get_current_qubits()
    assert qubits == [33, 22]


def test_generate_circuits(dummy_sb_instance):
    """Verify that SequentialBenchmark generates a dummy circuit for the initial run index."""
    dummy_sb_instance.generate_circuits()
    circuits = dummy_sb_instance.circuits
    assert len(circuits) == 1
    assert circuits[0].num_qubits == 2

    dummy_sb_instance.run_index = 1
    dummy_sb_instance.generate_circuits()
    circuits = dummy_sb_instance.circuits
    assert len(circuits) == 1
    assert circuits[0].num_qubits == 3


def test_should_stop(dummy_sb_instance):
    """Verify that SequentialBenchmark correctly decides whether it stops based on some results."""
    should_stop = dummy_sb_instance.should_stop({"max_qubit": 2})
    assert not should_stop
    should_stop = dummy_sb_instance.should_stop({"max_qubit": 3})
    assert should_stop
    should_stop = dummy_sb_instance.should_stop({"max_qubit": 4})
    assert should_stop


def test_run(dummy_sb_instance):
    """Verify that SequentialBenchmark runs sub-benchmarks sequentially and correctly stops."""
    device = IdealSimulator()
    dummy_sb_instance.generate_circuits()
    dummy_sb_instance.run(device)
    assert len(dummy_sb_instance.benchmarks) == 2
    assert len(dummy_sb_instance.all_results) == 2
    assert dummy_sb_instance.benchmarks[0].circuits[0].num_qubits == 2
    assert dummy_sb_instance.benchmarks[1].circuits[0].num_qubits == 3
    assert dummy_sb_instance.all_results[0]["max_qubit"] == 2
    assert dummy_sb_instance.all_results[1]["max_qubit"] == 3
    assert dummy_sb_instance.all_results[0]["parameter"] == "some value"
    assert dummy_sb_instance.all_results[1]["parameter"] == "some value"


@pytest.mark.parametrize("stopping_num", [2, 3, 4, 5])
def test_find_largest_successful_qubit(stopping_num, dummy_sb_instance):
    """Verify that SequentialBenchmark can correctly find the largest successful qubit."""
    device = IdealSimulator()
    dummy_sb_instance.set_stopping_num(stopping_num)
    dummy_sb_instance.generate_circuits()
    dummy_sb_instance.run(device)
    qubit = dummy_sb_instance.get_largest_successful_qubit()
    if stopping_num in [3, 4]:
        assert qubit == stopping_num - 1
    else:
        assert qubit is None
