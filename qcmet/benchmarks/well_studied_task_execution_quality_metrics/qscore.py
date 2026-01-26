"""QScore Metric.

This module implements the QScore benchmark for the QCMet framework. The metric
quantifies a quantum device's ability to solve combinatorial optimization
problems by running MaxCut instances on random graphs and comparing the device
solution quality to the exact optimum. The primary figure of merit is the
approximation ratio, which can be mapped into a device "QScore" at a given
problem size. Here the benchmarking procedure follows M5.2 from arxiv:2502.06717

TODO: The current implementation only runs a single instance for a fixed number of qubits.
To get the actual QScore metric, this needs to be run for inreasing numbers of qubits until
the largest size satisfying the pass criterion is reached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pathlib import Path

    from qcmet.core import FileManager
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from scipy.optimize import minimize
from tqdm import tqdm

from qcmet.benchmarks import BaseBenchmark


class QScoreSingleInstance(BaseBenchmark):
    """Implementation of the QScore (MaxCut-based) metric for a fixed number of qubits.

    This benchmark samples random graphs over `num_qubits`, constructs a QAOA-style
    circuit (alternating RZZ cost layers and RX mixer layers), runs on a target
    backend, and optimizes the variational parameters to minimize a MaxCut-derived
    cost. The final figure of merit is `beta`, an aggregated score, normalized against
    the optimal solution and a random solution, is computed from the observed costs
    across `n_graphs` instances; the benchmark is considered passed if `beta > 0.2`.

    Notes:
        - Random graphs are generated with Erdős-Rényi-style edge sampling where each
          possible edge is included independently with probability 1/2.
        - Variational parameter optimization uses COBYLA via `scipy.optimize.minimize`.
        - The cost uses a ±1/2 edge contribution convention to match the MaxCut scoring,
          followed by a shift by `len(graph)/2` to align with the reference baseline.
        - The `beta` aggregation follows the reference implementation's normalization
          (see comment in `_analyze`) using an `n^(3/2)` scaling.

    """

    def __init__(
        self,
        qubits: int | List[int],
        depth: int,
        n_graphs: int = 100,
        seed: int | None = None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the single QScore benchmark.

        Args:
            qubits (int | List[int]): Number of qubits as an integer or list of indices.
            depth (int): Number of QAOA layers (p), i.e., alternating cost/mixer blocks.
            n_graphs (int, optional): Number of random graph instances for benchmarking.
                Defaults to 100.
            seed (int | None, optional): Random seed for reproducibility. If None, a seed
                is drawn uniformly from [0, 1e8). Defaults to None.
            save_path (str | Path | FileManager | None, optional): Directory path to save results.
                Defaults to None.

        """
        super().__init__("QScoreSingle", qubits, save_path=save_path)
        self.config["depth"] = depth
        self.config["n_graphs"] = n_graphs

        if seed is None:
            seed = np.random.randint(100000000)

        self.config["seed"] = seed

        self.rng = np.random.default_rng(seed)

    def _generate_random_graph(self):
        """Generate a random graph with edge inclusion probability 1/2.

        Constructs an undirected simple graph over `self.num_qubits` vertices by
        independently sampling each possible edge (i, j), i != j, with probability 0.5.

        Returns:
            List[Tuple[int, int]]: List of edges represented as (v1, v2) integer pairs,
            with 0 ≤ v1 < v2 < self.num_qubits.

        """
        graph = []

        for v1 in range(self.num_qubits):
            for v2 in range(v1, self.num_qubits):
                if v1 != v2 and self.rng.random() < 0.5:
                    graph.append((v1, v2))

        return graph

    def _create_qaoa_circuit(self, graph):
        """Build a parameterized QAOA-style circuit for MaxCut on a given graph.

        Creates a circuit with `self.config["depth"]` layers. Each layer consists of:
            - Cost phase separators: RZZ gates for each edge (v1, v2) with angle
              `2 * gamma_{layer, edge}` implemented via `Parameter` objects.
            - Mixer unitaries: RX rotations on all qubits with angle `2 * beta_{layer, qubit}`.

        The circuit ends with measurement on all qubits.

        Args:
            graph (List[Tuple[int, int]]): Edge list defining the MaxCut instance.

        Returns:
            QuantumCircuit: A parameterized circuit with `num_parameters = depth * (|E| + |V|)`
            containing named Qiskit Parameter instances for all gammas and betas.

        """
        qc = QuantumCircuit(self.num_qubits)

        # Create layers of circuits
        for i in range(self.config["depth"]):
            for j, (v1, v2) in enumerate(graph):
                qc.rzz(2 * Parameter("gamma_{}_{}".format(i, j)), v1, v2)

            for j in range(self.num_qubits):
                qc.rx(2 * Parameter("beta_{}_{}".format(i, j)), j)

        qc.measure_all()

        return qc

    def _generate_circuits(self):
        """Generate the set of circuits required for the QScore benchmark.

        For each of the `n_graphs` instances:
            1. Sample a random graph using `_generate_random_graph`.
            2. Build a parameterized QAOA circuit via `_create_qaoa_circuit`.
            3. Create a concrete circuit by assigning all parameters to 1.0 (as an
               initial setting for execution).

        Returns:
            List[Dict]: A list of dictionaries, each containing:
                - 'circuit' (QuantumCircuit): Circuit with parameters assigned to 1.0.
                - 'circuit_unassigned' (QuantumCircuit): Parameterized template circuit.
                - 'graph' (List[Tuple[int, int]]): Edge list for this instance.

        Notes:
            - The initial parameter assignment is a heuristic; the optimization stage
              will subsequently update parameters per instance.
            - The number of generated circuits equals `self.config["n_graphs"]`.

        """
        circs = []
        for _ in range(self.config["n_graphs"]):
            graph = self._generate_random_graph()

            circ_unassigned = self._create_qaoa_circuit(graph)

            circ = circ_unassigned.assign_parameters(
                np.ones(circ_unassigned.num_parameters)
            )

            circs.append(
                {
                    "circuit": circ,
                    "circuit_unassigned": circ_unassigned,
                    "graph": graph,
                }
            )

        return circs

    def _compute_cost(self):
        """Compute the MaxCut-derived cost for each executed circuit instance.

        For each instance, this method:
            - Iterates over measured bitstrings and their counts.
            - Computes the edge-wise contribution: +1/2 if both endpoints share the
              same bit value, −1/2 if they differ (note the sign convention).
            - Applies a global shift of `−len(graph)/2` so that the energy corresponds to
              the number of cuts (times -1).
            - Aggregates the expectation over the empirical distribution and normalizes
              by the total number of shots.

        Updates:
            - Adds a column `cost_vals` to `self.experiment_data`, containing one
              scalar per instance (float).

        Returns:
            None: Operates in place on `self.experiment_data`.


        Notes:
            - The normalization uses `self._runtime_params["num_shots"]`.
            - The chosen ±1/2 convention differs from the definition in the original paper,
              but is necessary to obtain the intended score alignment for QScore.

        """
        cost_vals = []
        for i in range(len(self.experiment_data["circuit"])):
            total_cost = 0
            counts = self.experiment_data["circuit_measurements"][i]
            graph = self.experiment_data["graph"][i]
            for bit_string, count in counts.items():
                # TODO: implement more efficiently
                expectation = 0
                for x, y in graph:
                    # note that the factor of 1/2 is different from original equation in paper, but is required to get the right MaxCut score
                    if bit_string[x] != bit_string[y]:
                        expectation -= 1 / 2
                    else:
                        expectation += 1 / 2
                expectation -= len(graph) / 2
                total_cost += expectation * count
            cost_vals.append(total_cost / self._runtime_params["num_shots"])

        self.experiment_data["cost_vals"] = cost_vals

    def _optimize_parameters(self):
        """Optimize variational parameters per instance using COBYLA.

        For each circuit instance:
            - Extracts the parameterized circuit template (`circuit_unassigned`).
            - Defines a cost function that:
                (a) assigns candidate parameters,
                (b) executes the circuit via `_run_online`,
                (c) computes the cost via `_compute_cost`,
                (d) returns the scalar cost.
            - Runs `scipy.optimize.minimize` with method "COBYLA", starting from
              a randomly initialized parameter vector (normally distributed around 1).
            - Re-evaluates the final solution to store the resulting `cost_vals`.

        Updates:
            - Replaces `self._experiment_data` with the concatenation of per-instance
              results, reset to a clean index.

        Returns:
            None: Operates in place on `self._experiment_data` and `self.experiment_data`.


        Notes:
            - The optimization uses COBYLA, a derivative-free optimizer well-suited for noisy, non-smooth objectives.

        """
        all_circuits = self.experiment_data

        final_results = []

        print("Optimizing circuits...")
        for i in tqdm(range(len(self.experiment_data))):
            circ = all_circuits["circuit_unassigned"].iloc[i]
            self._experiment_data = all_circuits.iloc[[i]].copy()
            self._experiment_data.reset_index(drop=True, inplace=True)

            def cost_fun(pars, circ=circ):
                self.experiment_data["circuit"] = [circ.assign_parameters(pars)]
                self._run_online()
                self._compute_cost()
                return np.squeeze(self.experiment_data["cost_vals"])

            init_pars = self.rng.normal(loc=1.0, size=circ.num_parameters)
            res = minimize(cost_fun, x0=init_pars, method="COBYLA")

            cost_fun(res.x)

            final_results.append(self.experiment_data)

        self._experiment_data = pd.concat(final_results)
        self._experiment_data.reset_index(drop=True, inplace=True)

    def _analyze(self):
        """Run parameter optimization and compute the final beta value for the QScore metric.

        Orchestrates the end-to-end analysis:
            1) Optimizes variational parameters for all instances via `_optimize_parameters`.
            2) Uses the aggregated costs to compute `beta`:
               beta = sum_i [ (-cost_i - n(n-1)/8 ) / (0.178 * n^(3/2)) ] / n_graphs,
               where n = `self.num_qubits`. This follows the reference implementation’s
               scaling and normalization (see inline comment for details).
            3) Declares the benchmark as passed if `beta > 0.2`.

        Returns:
            Dict[str, object]: A result dictionary containing:
                - 'beta' (float): The aggregated beta value for the QScore for this size.
                - 'passed' (bool): True if `beta > 0.2`, else False.

        Notes:
            - The normalization term uses `n^(3/2)` with coefficient 0.178, and the
              baseline subtraction `(n(n-1))/8` matches the reference practice.
            - Printed diagnostics include per-instance costs and edge counts to aid
              sanity-checks.

        """
        self._optimize_parameters()

        beta = float(
            np.sum(
                (
                    -self.experiment_data["cost_vals"]  # the score
                    - (self.num_qubits * (self.num_qubits - 1))
                    / 8  # the reference from random solutions
                    # Note that original paper uses n**2 but actual reference implementation
                    # (https://github.com/myQLM/qscore/blob/master/qat/qscore/benchmark.py) includes this
                    # factor taking into account that no edge between a vertex and itself can ever exist
                )
                / (0.178 * self.num_qubits ** (3 / 2))
                / self.config["n_graphs"]
            )
        )

        return {"beta": beta, "passed": beta > 0.2}
