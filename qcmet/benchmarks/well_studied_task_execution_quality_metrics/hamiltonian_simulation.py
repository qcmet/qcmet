"""General Hamiltonian simulation benchmarks.

This module implements the base `HamiltonianSimulation` benchmarking class, providing the skeleton
to set up Hamiltonian simulation benchmarks in the QCMet framework. The resulting metric
evaluates how well a quantum computer can reproduce the populations for a given Trotterization
circuit and initial state preparation.

The concrete implementation for the 1D Fermi-Hubbard model follows the benchmarking
procedure M5.3 from arxiv:2502.06717.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pathlib import Path

    from qcmet.core import FileManager
import numpy as np
from qiskit import QuantumCircuit

from qcmet.benchmarks import BaseBenchmark
from qcmet.utils.fidelities import normalized_fidelity
from qcmet.utils.noiseless_simulation import compute_ideal_outputs


class HamiltonianSimulation(BaseBenchmark):
    """Benchmark class for simulating quantum dynamics of a Hamiltonian with Trotter evolution.

    This class implements a general Trotterized Hamiltonian simulation bencharking
    workflow, including estimating the final populations under a Trotterized evolution
    for an initial state a parameterized ansatz and evaluating the the normalized
    fidelity as the final metric.

    Concrete HamiltonianSimulation instances should either inherit from this base
    class and overwrite the _trotter_step function, or pass in the a single Trotter
    layer (as qiskit circuit) into the constructor.

    """

    def __init__(
        self,
        simulation_name: str,
        qubits: int | List[int],
        evolution_circuit: QuantumCircuit | None = None,
        init_circuit: QuantumCircuit | None = None,
        n_steps: int = 1,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the HamiltonianSimulation benchmark instance with configuration and circuits.

        Stores the evolution circuit (a single Trotter step), and initial circuit,
        if these are not passed in, the respective properties/functions should be
        overwritten in inheriting classes. Sets the number of steps for the application
        of the Trotter evolution ansatz.

        Args:
            simulation_name (str): Name of the Hamiltonian simulation.
            qubits (list[int]): List of qubit indices used in the simulation.
            evolution_circuit (QuantumCircuit, optional): Circuit representing one step of the system's evolution.
            init_circuit (QuantumCircuit, optional): Circuit for preparing the initial quantum state.
            n_steps (int, optional): Number of Trotter steps to apply. Defaults to 1.
            save_path (str | Path | FileManager | None, optional): Directory path to save results.
                Defaults to None.

        """
        super().__init__(simulation_name, qubits, save_path=save_path)
        self.config["n_steps"] = n_steps
        self._evolution_circuit = evolution_circuit
        self._init_circuit = init_circuit
        self._composed_circuit = None

    @property
    def evolution_circuit(self):
        """Return the composed quantum circuit with initial state, Trotter steps, and measurements.

        Returns:
            QuantumCircuit: The composed quantum circuit with initial state preparation,
            Trotter evolution, and measurement.

        """
        if self._composed_circuit is None:
            self._composed_circuit = self.initial_state

            for _ in range(self.config["n_steps"]):
                self._composed_circuit = self._trotter_step(self._composed_circuit)

        self._composed_circuit.measure_all()
        return self._composed_circuit

    def _generate_circuits(self):
        """Return the final circuit as list for benchmarking workflow.

        Returns:
            list[QuantumCircuit]: A list containing the full evolution circuit.

        """
        return [self.evolution_circuit]

    def _analyze(self):
        """Evaluate the normalized fidelities by comparing measured and ideal probabilities.

        Returns:
            dict: Dictionary containing the normalized fidelity (as single element list).

        """
        self.measurements_to_probabilities()

        fidelity = []
        meas_probs_all = []
        meas_probs_exact = []
        for i, circuit in enumerate(self.experiment_data["circuit"]):
            probabilities = self.experiment_data["meas_prob"].iloc[i]
            ideal_probabilities = compute_ideal_outputs(circuit)

            probs_circuit = np.zeros(len(ideal_probabilities))
            probs_ideal = np.zeros(len(ideal_probabilities))

            for j, bitstring in enumerate(ideal_probabilities.keys()):
                if bitstring in probabilities:
                    probs_circuit[j] = probabilities[bitstring]
                probs_ideal[j] = ideal_probabilities[bitstring]

            meas_probs_all.append(
                dict(zip(ideal_probabilities.keys(), probs_ideal, strict=True))
            )
            meas_probs_exact.append(
                dict(zip(ideal_probabilities.keys(), probs_circuit, strict=True))
            )

            fidelity.append(normalized_fidelity(probs_ideal, probs_circuit))

        self.experiment_data["exact_probs"] = meas_probs_exact
        self.experiment_data["meas_probs_extended"] = meas_probs_all
        self.experiment_data["normalized_fidelities"] = fidelity
        return {"normalized_fidelity": fidelity}

    def _plot(self, axes):
        """Plot exact and measured probability distributions.

        Args:
            axes (matplotlib.axes.Axes): Axes to draw the bar charts on.

        Returns:
            matplotlib.legend.Legend: Legend for the plotted bars.

        """
        bitstrings = self.experiment_data["exact_probs"][0].keys()
        xlabels = [rf"$\vert{i}\rangle$" for i in bitstrings]
        x_range = np.arange(0, 2**self.num_qubits, dtype=int)
        axes.bar(
            x_range,
            [self.experiment_data["exact_probs"][0][key] for key in bitstrings],
            label="Exact probabilities",
            edgecolor="black",
            color="none",
        )
        axes.bar(
            x_range,
            [self.experiment_data["meas_probs_extended"][0][key] for key in bitstrings],
            width=0.5,
            label=f"{self._runtime_params['device'].name} results",
            color="black",
            alpha=0.6,
        )
        axes.set_xticks(x_range, xlabels, rotation=45, fontsize=10)
        axes.set_xlabel(r"$x$")
        axes.set_ylabel(r"$p_x$")
        return axes.legend()

    def _trotter_step(self, circuit):
        """Apply a single Trotter evolution circuit to the given quantum circuit.

        Args:
            circuit (QuantumCircuit): The circuit to which the Trotter step is applied.

        Returns:
            QuantumCircuit: The updated circuit with the evolution circuit applied.

        """
        assert self._evolution_circuit is not None

        return circuit.compose(self._evolution_circuit)

    @property
    def initial_state(self):
        """Return the initial quantum circuit for state preparation.

        Returns:
            QuantumCircuit: The initial state circuit or a default empty circuit.

        """
        if self._init_circuit is None:
            return QuantumCircuit(self.num_qubits)
        else:
            return self._init_circuit
