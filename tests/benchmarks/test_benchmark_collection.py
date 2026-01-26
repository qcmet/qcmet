"""test_benchmark_collection.py.

Unit tests for the BenchmarkCollection in qcmet.utils.benchmark_collection.
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import pytest
from qiskit import QuantumCircuit

from qcmet import BenchmarkCollection
from qcmet.benchmarks import BaseBenchmark
from qcmet.devices import IdealSimulator


class DummyBenchmarkWithPlotting(BaseBenchmark):
    """Concrete BaseBenchmark with plotting for testing BenchmarkCollection methods."""

    def _generate_circuits(self) -> List[QuantumCircuit] | Dict[str, any]:
        c = QuantumCircuit(1)
        c.x(0)
        c.measure_all()
        return [c]

    def _analyze(self) -> Dict[str, any]:
        return self._experiment_data["circuit_measurements"].iloc[0]

    def _plot(self, axes):
        return axes

    def save(self):
        """Mock save function."""
        self._experiment_data = {}
        self._experiment_data["saved"] = True


class DummyBenchmark2Qubits(BaseBenchmark):
    """Concrete BaseBenchmark without plotting for testing BenchmarkCollection methods."""

    def _generate_circuits(self) -> List[QuantumCircuit] | Dict[str, any]:
        c = QuantumCircuit(2)
        c.x(0)
        c.x(1)
        c.measure_all()
        return [c, c]

    def _analyze(self) -> Dict[str, any]:
        c0 = self._experiment_data["circuit_measurements"].iloc[0]
        c1 = self._experiment_data["circuit_measurements"].iloc[1]
        return {"circuit0": c0, "circuit1": c1}

    def save(self):
        """Mock save function."""
        self._experiment_data = {}
        self._experiment_data["saved"] = True


@pytest.fixture
def benchmark_collection_instance(request):
    """Fixture to create a benchmark collection containing two dummies."""
    dummy1 = DummyBenchmarkWithPlotting("Dummy1", 1)
    dummy2 = DummyBenchmark2Qubits("Dummy2", 2)
    benchmark_collection = BenchmarkCollection([dummy1, dummy2])
    if request.cls is not None:
        request.cls.benchmark_collection = benchmark_collection
    return benchmark_collection


def test_create_benchmark_labels(benchmark_collection_instance):
    """Verify that the labels of benchmarks are generated correctly."""
    assert benchmark_collection_instance._benchmark_labels == [
        "Benchmark0_Dummy1",
        "Benchmark1_Dummy2",
    ]
    benchmark_collection2 = BenchmarkCollection(
        {
            "label1": DummyBenchmarkWithPlotting("Dummy3", 1),
            "label2": DummyBenchmark2Qubits("Dummy4", 2),
        }
    )
    assert benchmark_collection2._benchmark_labels == ["label1", "label2"]


def test_num_qubits(benchmark_collection_instance):
    """Verify that the number of qubits for the collection is returned as a list."""
    assert benchmark_collection_instance.num_qubits == {
        "Benchmark0_Dummy1": 1,
        "Benchmark1_Dummy2": 2,
    }


def test_generate_circuits(benchmark_collection_instance):
    """Verify that the generated circuits contains all circuits from all sub-benchmarks."""
    circuits = benchmark_collection_instance._generate_circuits()
    assert len(circuits) == 3


def test_run_with_universal_shots(benchmark_collection_instance):
    """Verify that providing integer num_shots makes all circuits run with the same number of shots."""
    device = IdealSimulator()
    benchmark_collection_instance.generate_circuits()
    benchmark_collection_instance.run(device, num_shots=100)
    assert benchmark_collection_instance._runtime_params["num_shots"] == 100
    for counts in benchmark_collection_instance._experiment_data[
        "circuit_measurements"
    ]:
        assert sum(counts.values()) == 100


def test_run_with_different_shots(benchmark_collection_instance):
    """Verify that providing a list to num_shots runs circuits with different number of shots."""
    device = IdealSimulator()
    benchmark_collection_instance.generate_circuits()
    benchmark_collection_instance.run(device, num_shots=[100, 200])
    assert benchmark_collection_instance._runtime_params is None
    for i, benchmark in enumerate(benchmark_collection_instance._benchmarks):
        for counts in benchmark._experiment_data["circuit_measurements"]:
            assert sum(counts.values()) == 100 * (i + 1)


def test_run_with_wrong_shots_list(benchmark_collection_instance):
    """Verify that run correctly throws an error if provided with mismatching list of shots."""
    device = IdealSimulator()
    benchmark_collection_instance.generate_circuits()
    with pytest.raises(ValueError):
        benchmark_collection_instance.run(device, [100, 200, 300])


def test_analyze_with_universal_shots(benchmark_collection_instance):
    """Verify that analyze works as expected with universal shots by distributing the outcomes to sub-benchmarks."""
    device = IdealSimulator()
    benchmark_collection_instance.generate_circuits()
    benchmark_collection_instance.run(device, num_shots=100)
    results = benchmark_collection_instance.analyze()
    assert results is not None
    assert results["Benchmark0_Dummy1"]["1"] == 100
    assert results["Benchmark1_Dummy2"]["circuit0"]["11"] == 100
    assert results["Benchmark1_Dummy2"]["circuit1"]["11"] == 100


def test_analyze_with_different_shots(benchmark_collection_instance):
    """Verify that analyze works as expected with different shots."""
    device = IdealSimulator()
    benchmark_collection_instance.generate_circuits()
    benchmark_collection_instance.run(device, num_shots=[100, 200])
    results = benchmark_collection_instance.analyze()
    assert results is not None
    assert results["Benchmark0_Dummy1"]["1"] == 100
    assert results["Benchmark1_Dummy2"]["circuit0"]["11"] == 200
    assert results["Benchmark1_Dummy2"]["circuit1"]["11"] == 200


def test_has_plotting(benchmark_collection_instance):
    """Verify that has_plotting works correctly."""
    assert benchmark_collection_instance.has_plotting() is True
    benchmark_collection2 = BenchmarkCollection(
        [DummyBenchmark2Qubits("Dummy3", 2), DummyBenchmark2Qubits("Dummy4", 2)]
    )
    assert benchmark_collection2.has_plotting() is False


def test_plot_without_axes():
    """Verify that plot correctly creates axes and plot for each sub-benchmark."""
    benchmark_collection = BenchmarkCollection(
        {
            "label1": DummyBenchmarkWithPlotting("Dummy1", 1),
            "label2": DummyBenchmarkWithPlotting("Dummy2", 1),
            "label3": DummyBenchmark2Qubits("Dummy3", 2),
            "label4": DummyBenchmarkWithPlotting("Dummy4", 1),
        }
    )
    benchmark_collection.plot()
    axes = plt.gcf().axes
    assert len(axes) == 3
    assert axes[0].get_title() == benchmark_collection._benchmark_labels[0]
    assert axes[1].get_title() == benchmark_collection._benchmark_labels[1]
    assert axes[2].get_title() == benchmark_collection._benchmark_labels[3]


def test_plot_with_axes():
    """Verify that plot correctly puts sub-benchmark plots into given axes."""
    benchmark_collection = BenchmarkCollection(
        {
            "label1": DummyBenchmarkWithPlotting("Dummy1", 1),
            "label2": DummyBenchmarkWithPlotting("Dummy2", 1),
            "label3": DummyBenchmark2Qubits("Dummy3", 2),
            "label4": DummyBenchmarkWithPlotting("Dummy4", 1),
        }
    )
    _, axes = plt.subplots(ncols=3)
    benchmark_collection.plot(axes=[axes[1], axes[2], axes[0]])
    assert axes[1].get_title() == benchmark_collection._benchmark_labels[0]
    assert axes[2].get_title() == benchmark_collection._benchmark_labels[1]
    assert axes[0].get_title() == benchmark_collection._benchmark_labels[3]


def test_plot_with_wrong_axes_list(benchmark_collection_instance):
    """Verify that plot correctly throws an error if provided with mismatching list of axes."""
    _, axes = plt.subplots(ncols=10)
    with pytest.raises(ValueError):
        benchmark_collection_instance.plot(axes)


def test_save(benchmark_collection_instance):
    """Verify that save correctly calls the save function of each sub-benchmark."""
    benchmark_collection_instance.set_save_path("test")
    benchmark_collection_instance.save()
    assert (
        benchmark_collection_instance._benchmarks[0]._experiment_data["saved"] is True
    )
    assert (
        benchmark_collection_instance._benchmarks[1]._experiment_data["saved"] is True
    )
