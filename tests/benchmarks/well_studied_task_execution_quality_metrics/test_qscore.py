"""test_qscore.py.

Unit tests for the QScore benchmark in qcmet.
"""
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit

from qcmet import IdealSimulator
from qcmet.benchmarks import QScore, QScoreSingleInstance


def test_random_graph_reproducibility_and_structure():
    """Verify random graph generation is reproducible and structurally valid.

    - With a fixed seed, two instances produce identical edge lists.
    - No self-loops: (v, v) is never present.
    - Edges are stored as ordered pairs with v1 < v2.
    """
    seed = 12345
    qs1 = QScoreSingleInstance(qubits=5, depth=1, n_graphs=1, seed=seed)
    qs2 = QScoreSingleInstance(qubits=5, depth=1, n_graphs=1, seed=seed)

    g1 = qs1._generate_random_graph()
    g2 = qs2._generate_random_graph()

    assert g1 == g2, "Graphs should be identical with the same seed"
    for v1, v2 in g1:
        assert v1 != v2, "Self-loops must not be present"
        assert v1 < v2, "Edges should be stored with v1 < v2 ordering"
        assert 0 <= v1 < qs1.num_qubits and 0 <= v2 < qs1.num_qubits


def test_create_qaoa_circuit_parameters_and_measurements():
    """Check QAOA circuit structure, parameter count, naming, and measurements.

    - For a given graph and depth, the number of Parameters equals depth*(|E|+|V|).
    - Parameter names include 'gamma_{layer}_{edge_idx}' and 'beta_{layer}_{qubit_idx}'.
    - A measurement is added to each qubit.
    """
    qs = QScoreSingleInstance(qubits=3, depth=2, n_graphs=1, seed=42)
    # A simple graph: edges (0,1), (1,2)
    graph = [(0, 1), (1, 2)]

    qc = qs._create_qaoa_circuit(graph)
    assert isinstance(qc, QuantumCircuit)

    expected_num_parameters = qs.config["depth"] * (len(graph) + qs.num_qubits)
    assert qc.num_parameters == expected_num_parameters

    # Check parameter name patterns roughly (not exact set, but presence)
    param_names = {str(p) for p in qc.parameters}
    assert any(name.startswith("gamma_0_") for name in param_names), (
        "Gamma parameters for layer 0 missing"
    )
    assert any(name.startswith("gamma_1_") for name in param_names), (
        "Gamma parameters for layer 1 missing"
    )
    assert any(name.startswith("beta_0_") for name in param_names), (
        "Beta parameters for layer 0 missing"
    )
    assert any(name.startswith("beta_1_") for name in param_names), (
        "Beta parameters for layer 1 missing"
    )

    # Measurement count should equal number of qubits
    ops = qc.count_ops()
    assert ops.get("measure", 0) == qs.num_qubits


def test_generate_circuits_structure_and_length():
    """Validate output of _generate_circuits.

    - Returns a list of length n_graphs.
    - Each entry contains 'circuit', 'circuit_unassigned', and 'graph'.
    - 'circuit' is a QuantumCircuit with parameters assigned (no free Parameters).
    - 'circuit_unassigned' is a parameterized template (with Parameters).
    """
    qs = QScoreSingleInstance(qubits=4, depth=1, n_graphs=3, seed=7)
    circuits = qs._generate_circuits()

    assert isinstance(circuits, list)
    assert len(circuits) == qs.config["n_graphs"]

    for entry in circuits:
        assert set(entry.keys()) == {"circuit", "circuit_unassigned", "graph"}

        circ = entry["circuit"]
        circ_unassigned = entry["circuit_unassigned"]
        graph = entry["graph"]

        assert isinstance(circ, QuantumCircuit)
        assert isinstance(circ_unassigned, QuantumCircuit)
        assert isinstance(graph, list)

        # Assigned circuit should have no free parameters
        assert circ.num_parameters == 0
        # Unassigned circuit should have depth * (|E| + |V|) parameters
        expected_params = qs.config["depth"] * (len(graph) + qs.num_qubits)
        assert circ_unassigned.num_parameters == expected_params

        # Measurement present on all qubits
        ops = circ.count_ops()
        assert ops.get("measure", 0) == qs.num_qubits


def test_compute_cost_manual_counts_single_edge():
    """Confirm _compute_cost produces expected values for a simple case.

    For a graph with a single edge (0,1) and counts {'01': shots}, the contribution is:
      - Per shot: edge differs => -1/2; global shift -len(graph)/2 = -1/2 => total -1.
      - Averaged over shots: cost_vals should be -1.0.
    """
    qs = QScoreSingleInstance(qubits=2, depth=1, n_graphs=1, seed=99)
    qs.generate_circuits()

    # Prepare a minimal experiment_data by hand
    qs._experiment_data = qs.experiment_data.iloc[:0].copy()  # clear
    qs._experiment_data.loc[0, "circuit"] = [None]  # placeholder; only length is used
    qs._experiment_data.loc[0, "graph"] = [(0, 1)]
    qs._experiment_data["circuit_measurements"] = [{"01": 100}]

    # Set runtime shots used for normalization
    qs._runtime_params = {"num_shots": 100}

    qs._compute_cost()
    assert "cost_vals" in qs.experiment_data.columns
    assert pytest.approx(qs.experiment_data["cost_vals"].iloc[0], abs=1e-12) == -1.0


def test_optimize_parameters_runs_and_updates_costs():
    """Ensure _optimize_parameters executes and populates final cost values.

    - Use a small instance (n_graphs=1, qubits=3, depth=1) for speed.
    - Provide a backend via `run()` so `_run_online` has runtime params.
    - After optimization, `_experiment_data` contains `cost_vals`.
    """
    qs = QScoreSingleInstance(qubits=3, depth=1, n_graphs=1, seed=11)
    qs.generate_circuits()  # populates qs.experiment_data internally
    sim = IdealSimulator()

    # Run once to attach backend and shots to runtime params
    qs.run(sim, num_shots=64)

    # Execute the optimizer
    qs._optimize_parameters()

    assert hasattr(qs, "_experiment_data")
    assert "cost_vals" in qs._experiment_data.columns
    assert len(qs._experiment_data["cost_vals"]) == qs.config["n_graphs"]
    # Cost should be a finite float
    assert np.isfinite(float(qs._experiment_data["cost_vals"].iloc[0]))
    # Cost should be smaller than 0
    assert float(qs._experiment_data["cost_vals"].iloc[0]) < 0.0


def test_analyze_outputs_beta_and_passed_flag():
    """End-to-end test: optimize parameters, compute beta, and return result.

    - Uses IdealSimulator and small sizes for speed.
    - Verifies result dict contains 'beta' (float) and 'passed' (bool).
    """
    qs = QScoreSingleInstance(qubits=3, depth=1, n_graphs=1, seed=123)
    qs.generate_circuits()
    sim = IdealSimulator()
    qs.run(sim, num_shots=64)

    result = qs.analyze()

    assert isinstance(result, dict)
    assert "beta" in result and "passed" in result
    assert isinstance(result["beta"], float)
    assert result["beta"] > 0.0


def test_qscore_sequential_analyze():
    """Verify that QScore._analyze correctly outputs the QScore value or None in the resulting dict."""
    qscore = QScore(2, 100)

    with patch("qcmet.benchmarks.SequentialBenchmark.get_largest_successful_qubit") as mock_qubit:
        mock_qubit.return_value = None
        assert qscore._analyze()["QScore"] is None

    with patch("qcmet.benchmarks.SequentialBenchmark.get_largest_successful_qubit") as mock_qubit:
        mock_qubit.return_value = 4
        assert qscore._analyze()["QScore"] == 4


def test_qscore_sequential_plot():
    """Verify that QScore._plot correctly plots the QScore values of each QScoreSingleInstance."""
    device = IdealSimulator()
    qscore = QScore(2, 3)
    qscore.all_results = [{"beta": 0.4, "passed": True},
                          {"beta": 0.3, "passed": True}]
    qscore._runtime_params = {"device": device}
    _, ax = plt.subplots()
    qscore.plot(ax)
    line = ax.lines[0]
    assert (line.get_xdata() == [2, 3]).all()
    assert (line.get_ydata() == [0.4, 0.3]).all()
