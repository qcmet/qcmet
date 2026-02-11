"""Tests for the BaseBenchmark class of QCMet."""

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from qiskit import QuantumCircuit

from qcmet.benchmarks import BaseBenchmark
from qcmet.core import FileManager
from qcmet.devices import BaseDevice, IdealSimulator


class DummyBenchmark(BaseBenchmark):
    """Minimal concrete subclass for testing BaseBenchmark methods."""

    def _generate_circuits(self):
        return []

    def _analyze(self):
        return {}


class CircuitBenchmark(BaseBenchmark):
    """Concrete subclass that generates simple circuits."""

    def _generate_circuits(self):
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        return [qc1, qc2]

    def _analyze(self):
        return {"dummy": True}


class PlotBenchmark(BaseBenchmark):
    """Subclass that implements _plot."""

    def _generate_circuits(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        return [qc]

    def _analyze(self):
        return {}

    def _plot(self, ax):
        ax.plot([0, 1], [1, 0])
        ax.set_title("Test Plot")


class CompleteBenchmark(BaseBenchmark):
    """Subclass that implements _generate_circuits, _run_online, _analyze and _plot."""

    def _generate_circuits(self):
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc1.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()
        return [qc1, qc2]

    def _run_online(self):
        return {}

    def _analyze(self):
        return {"dummy": True}

    def _plot(self, axes):
        axes.plot([0, 1], [1, 0])
        axes.set_title("Test Plot")


class MaxJobsBenchmark(BaseBenchmark):
    """Subclass that implements _generate_circuits. _run and _analyze."""

    def _generate_circuits(self):
        qc1 = QuantumCircuit(1)
        qc1.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()
        return [qc1, qc2]

    def _run(self):
        return {}

    def _analyze(self):
        return {"dummy": True}


@pytest.fixture
def basic_benchmark():
    """Provide a fresh DummyBenchmark with minimal setup.

    Returns:
        DummyBenchmark: A benchmark instance with name 'test' and 1 qubit.

    """
    return DummyBenchmark(name="test", qubits=1)


@pytest.fixture
def tmp_file_manager(tmp_path):
    """Provide a FileManager pointed at a temporary directory.

    Args:
        tmp_path (Path): Built-in pytest fixture for temporary filesystem path.

    Returns:
        FileManager: Initialized FileManager using tmp_path.

    """
    return FileManager("test_bench", tmp_path)


@pytest.fixture
def benchmark_with_data():
    """Create a DummyBenchmark with initialized experiment_data DataFrame."""
    bench = DummyBenchmark(name="test", qubits=1)
    # Minimal circuit placeholder
    bench._experiment_data = pd.DataFrame({"id": ["abc"], "circuit": [None]})
    return bench


def test_load_circuit_measurements_with_dicts(benchmark_with_data):
    """Test loading simple measurement dicts into the experiment dataframe."""
    bench = benchmark_with_data
    measurements = {"00": 23, "11": 77}
    bench.load_circuit_measurements(measurements)

    assert "circuit_measurements" in bench._experiment_data.columns
    assert bench._experiment_data["circuit_measurements"].iloc[0] == {
        "00": 23,
        "11": 77,
    }


def test_load_circuit_measurements_with_counters(benchmark_with_data):
    """Test loading Counter objects into the experiment dataframe."""
    bench = benchmark_with_data
    measurements = [Counter({"10": 15, "01": 85})]
    bench.load_circuit_measurements(measurements)

    assert isinstance(bench._experiment_data["circuit_measurements"].iloc[0], Counter)
    assert bench._experiment_data["circuit_measurements"].iloc[0]["01"] == 85


def test_init_with_int_qubits():
    """Verify that initializing with an integer number of qubits sets attributes correctly."""
    bench = DummyBenchmark(name="init_test", qubits=3)
    assert isinstance(bench.qubits, list)
    assert bench.num_qubits == 3
    assert bench.save_enabled is False
    assert bench.file_manager is None


def test_init_with_negative_int_raises():
    """Ensure that a negative qubit count raises a ValueError."""
    with pytest.raises(ValueError):
        DummyBenchmark(name="neg_qubits", qubits=-1)


def test_init_with_list_qubits_and_invalid_list():
    """Test initialization with a valid list of qubit indices and with invalid entries."""
    bench = DummyBenchmark(name="qubit_indices_check", qubits=[2, 5, 7])
    assert bench.qubits == [2, 5, 7]
    assert bench.num_qubits == 3
    with pytest.raises(ValueError):
        DummyBenchmark(name="invalid_qubits_check", qubits=[0, "a", 1])


def test_init_with_save_path_variants(tmp_path, tmp_file_manager):
    """Verify save_path handling for str, Path, and FileManager inputs."""
    bench_str = DummyBenchmark(
        name="save_test_path", qubits=1, save_path=tmp_path / "outdir"
    )
    assert bench_str.save_enabled
    assert isinstance(bench_str.file_manager, FileManager)

    bench_path = DummyBenchmark(
        name="save_test_str", qubits=1, save_path=f"{str(tmp_path)}/outdir2"
    )
    assert bench_path.save_enabled
    assert isinstance(bench_path.file_manager, FileManager)

    fm = tmp_file_manager
    bench_fm = DummyBenchmark(name="save_test_file_manager", qubits=1, save_path=fm)
    assert bench_fm.save_enabled
    assert bench_fm.file_manager is fm


def test_has_plotting_and_default_plot(capsys):
    """Check that has_plotting() is False by default and plot() prints awarning."""
    bench = DummyBenchmark(name="plotting_test", qubits=1)
    assert not bench.has_plotting()

    result = bench.plot()
    captured = capsys.readouterr()
    assert "No plotting implemented" in captured.out
    assert result is None


def test_plot_with_override_returns_fig_ax():
    """Ensure a subclass override of _plot returns a figure and axes with expected content."""
    bench = PlotBenchmark(name="plotting_test", qubits=1)
    assert bench.has_plotting()

    # define plot function
    fig, ax = bench.plot()
    assert fig.axes
    assert ax.get_title() == "Test Plot"
    plt.close(fig)


def test_experiment_data_setter_and_circuits_getter():
    """Validate setting experiment_data via circuits and retrieving them via the circuits property."""
    bench = CircuitBenchmark(name="exp_data_test", qubits=1)
    qc = QuantumCircuit(1)
    qc.h(0)

    # setting with list of QuantumCircuits
    bench.experiment_data = [qc]
    df = bench.experiment_data
    assert isinstance(df, pd.DataFrame)
    assert "circuit" in df.columns

    circs = bench.circuits
    assert isinstance(circs, list)
    assert isinstance(circs[0], QuantumCircuit)

    # getter raises if no data
    bench2 = DummyBenchmark(name="exp_data_test_2", qubits=1)
    with pytest.raises(AttributeError):
        _ = bench2.circuits


def test_experiment_data_setter_rejects_bad_inputs():
    """Test checks if experiment_data setter rejects invalid formats."""
    bench = DummyBenchmark(name="bad_inputs_exp_data", qubits=1)

    with pytest.raises(ValueError):
        bench.experiment_data = "not a list"

    with pytest.raises(ValueError):
        bench.experiment_data = []

    with pytest.raises(ValueError):
        bench.experiment_data = [{"abc": 1}]


def test_generate_circuits_and_save(monkeypatch, tmp_path):
    """Test generate_circuits sets experiment_data and calls save() when enabled."""
    bench = CircuitBenchmark(
        name="test_generate_circuits_and_save", qubits=1, save_path=tmp_path
    )
    called = {"save": False}

    def fake_save():
        called["save"] = True

    monkeypatch.setattr(bench, "save", fake_save)
    bench.generate_circuits()

    assert isinstance(bench.experiment_data, pd.DataFrame)
    assert called["save"]


def test_circ_with_metadata_dict_and_circs_to_df():
    """Verify that _circ_with_metadata_dict and _circs_to_df produce correct DataFrame entries."""
    bench = CircuitBenchmark(name="metadata_dict_tester", qubits=2)
    qc = QuantumCircuit(2)
    qc.x(1)
    rec = bench._circ_with_metadata_dict(qc, extra=42)

    assert rec["circuit"] is qc
    assert isinstance(rec["id"], str)
    assert isinstance(rec["hash"], str)
    assert rec["extra"] == 42

    df = bench._circs_to_df([rec])
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["circuit"] is qc


def test_measurements_to_probabilities_single_row(basic_benchmark):
    """Check single-row measurement counts convert to correct probabilities."""
    bench = basic_benchmark
    bench._runtime_params = {"num_shots": 100}
    bench._experiment_data = pd.DataFrame(
        {"circuit_measurements": [{"0": 20, "1": 80}]}
    )
    bench.measurements_to_probabilities()

    assert "meas_prob" in bench.experiment_data.columns
    probs = bench.experiment_data["meas_prob"].iloc[0]
    assert probs["0"] == pytest.approx(0.20)
    assert probs["1"] == pytest.approx(0.80)


def test_measurements_to_probabilities_multiple_rows(basic_benchmark):
    """Check multi-row measurement counts convert to correct normalized probabilities."""
    bench = basic_benchmark
    bench._runtime_params = {"num_shots": 4}
    bench._experiment_data = pd.DataFrame(
        {
            "circuit_measurements": [
                {"00": 1, "11": 3},
                {"01": 2, "10": 2},
            ]
        }
    )
    bench.measurements_to_probabilities()

    df = bench.experiment_data
    first = df["meas_prob"].iloc[0]
    assert first == {"00": 0.25, "11": 0.75}
    second = df["meas_prob"].iloc[1]
    assert second == {"01": 0.50, "10": 0.50}


def test_measurements_to_probabilities_zero_shots_raises(basic_benchmark):
    """Verify that zero shots triggers a ZeroDivisionError."""
    bench = basic_benchmark
    bench._runtime_params = {"num_shots": 0}
    bench._experiment_data = pd.DataFrame({"circuit_measurements": [{"0": 1}]})
    with pytest.raises(ZeroDivisionError):
        bench.measurements_to_probabilities()


def test_measurements_to_probabilities_missing_runtime_params_raises(basic_benchmark):
    """Verify that missing runtime_params causes an AttributeError."""
    bench = basic_benchmark
    bench._experiment_data = pd.DataFrame({"circuit_measurements": [{"0": 1}]})
    with pytest.raises(AttributeError):
        bench.measurements_to_probabilities()


def test_qubit_routing():
    """Check that instructions are assigned to the correct qubit."""
    bench = CircuitBenchmark(name="exp_data_test", qubits=[4])
    bench.generate_circuits()
    circ1 = bench.circuits[0]
    circ2 = bench.circuits[1]
    assert any(
        ci.operation.name == "h" and 4 in [circ1.find_bit(q).index for q in ci.qubits]
        for ci in circ1.data
    )
    assert not any(
        any(circ1.find_bit(q).index in [0, 1, 2, 3] for q in ci.qubits)
        for ci in circ1.data
    )
    assert any(
        ci.operation.name == "x" and 4 in [circ2.find_bit(q).index for q in ci.qubits]
        for ci in circ2.data
    )
    assert not any(
        any(circ1.find_bit(q).index in [0, 1, 2, 3] for q in ci.qubits)
        for ci in circ2.data
    )


def test_call():
    """Verify call function returns correct results and creates a plot with correct axes title."""
    experiment = CompleteBenchmark(name="test", qubits=1)
    fig, ax = plt.subplots()
    results = experiment(device=BaseDevice, axes=ax)
    assert ax.get_title() == "Test Plot"
    assert results == {"dummy": True}


def test_max_circs_per_job():
    """Verify that max_circs_per_job returns expected number of measurements in correct order."""
    experiment = MaxJobsBenchmark(name="test", qubits=1)
    experiment.generate_circuits()
    experiment.run(device=IdealSimulator(), num_shots=100, circs_per_job=1)
    assert len(experiment.experiment_data["circuit_measurements"])
    assert experiment.experiment_data.loc[0, "circuit_measurements"]["0"] == 100
    assert experiment.experiment_data.loc[1, "circuit_measurements"]["1"] == 100
