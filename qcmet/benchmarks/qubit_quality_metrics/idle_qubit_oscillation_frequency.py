"""Idle qubit oscillation frequency metric.

This module implements a benchmark for the frequency at which an idle
qubit periodically losing and regaining coherence. This metrics quantifies
the effect of non-Markovian noise induced coherence revivals. Here the
benchmarking procedure follows M2.3 from arxiv:2502.06717.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from qcmet.core import FileManager
import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit

from qcmet.benchmarks import BaseBenchmark


class IdleQubitOscillationFrequency(BaseBenchmark):
    """Implementation of the idle qubit oscillation frequency metric.

    This class generates idle circuits for three initial states, |0>, |+> and |R>,
    measures the output under three bases, Z, X and Y, and computes
    the qubit purity oscillation frequency for each initial state, giving the
    largest among the three as the metric.
    """

    def __init__(
        self,
        dt: float,
        t_max: float,
        extra_zz_crosstalk: float = 0,
        qubit_index: int = 0,
        crosstalk_qubit_index=None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the idle qubit oscillation frequency metric.

        Args:
            dt (float): The time interval for each idle step.
            t_max (float): The max idle time.
            extra_zz_crosstalk (float, optional): The strength of extra ZZ gates for simulating
                non-Markovian noise. If non-zero, circuits generated will be two-qubit circuits
                containing extra ZZ gates for each idle step. Defaults to 0.
            qubit_index (int, optional): The qubit index for routing. Defaults to 0.
            crosstalk_qubit_index (optional): The qubit index for the extra qubit used when extra_zz_crosstalk > 0.
                Defaults to qubit_index + 1.
            save_path (str, optional): Directory path to save results. Defaults to None.

        """
        if extra_zz_crosstalk == 0:
            qubits = [qubit_index]
        else:
            if crosstalk_qubit_index is None:
                crosstalk_qubit_index = qubit_index + 1
            qubits = [qubit_index, crosstalk_qubit_index]

        super().__init__(
            "IdleQubitOscillationFrequency", qubits=qubits, save_path=save_path
        )
        self.config["dt"] = dt
        self.config["t_max"] = t_max
        self.config["extra_zz_crosstalk"] = extra_zz_crosstalk
        self.config["steps"] = int(t_max // dt)
        self.config["idle_times"] = np.arange(
            0, self.config["t_max"], self.config["dt"]
        )[: self.config["steps"]]
        self._Z = 0
        self._X = 1
        self._Y = 2
        self._bases = [self._Z, self._X, self._Y]
        self._basis_names = ["Z", "X", "Y"]
        self._initial_state_names = ["0", "+", "R"]
        self._fit_results = []
        self._purities = []

    def prepare_initial_state(self, circuit, initial_state):
        """Prepare a specified initial state on `circuit`.

        Args:
            circuit (QuantumCircuit): An empty circuit.
            initial_state (int): The initial state, represented by `self._Z`, `self._X` or `self._Y`.

        Returns:
            None. The method updates `circuit` in place.

        """
        if initial_state == self._X:
            circuit.h(0)
        elif initial_state == self._Y:
            circuit.rx(-np.pi / 2, 0)
        if self.config["extra_zz_crosstalk"] != 0:
            circuit.h(1)

    def add_idle_gates(self, circuit, steps):
        """Add a number of idle gates to `circuit` corresponding to `steps`.

        Args:
            circuit (QuantumCircuit): A circuit to add idle gates to.
            steps (int): The number of idle steps.

        Returns:
            None. The method updates `circuit` in place.

        """
        for _ in range(steps):
            circuit.id(0)
            if self.config["extra_zz_crosstalk"] != 0:
                circuit.rzz(self.config["extra_zz_crosstalk"], 0, 1)
            circuit.barrier()

    def change_measurement_basis(self, circuit, basis):
        """Change the measurement basis of `circuit` to be in `basis`.

        Args:
            circuit (QuantumCircuit): A circuit to change the measurement basis for.
            basis (int): The basis to change to, represented by `self._Z`, `self._X` or `self._Y`.

        Returns:
            None. The method updates `circuit` in place.

        """
        if basis == self._X:
            circuit.h(0)
        elif basis == self._Y:
            circuit.sdg(0)
            circuit.h(0)

    def _generate_circuits(self):
        """Generate circuits for measuring purity oscillation.

        For each initial state in {|0>, |+>, |R>}, and each basis in {Z, X, Y},
        each circuit is built with the following steps:
            1. Use corresponding gates to prepare the initial state.
            2. Apply a sequence of idle gates (identity gates).
            3. Rotate the qubit with the corresponding basis changing gate(s).
            4. Measure the qubit.

        If `self.config["extra_zz_crosstalk"]` is 0, circuits will be one-qubit circuits.
        Otherwise, an extra ZZ gate will be added after each idle gate to simulate non-Markovian
        noise, and the circuits become two-qubit circuits. In this case, only qubit 0 is measured.

        Returns:
            list: A list of QuantumCircuit objects with varying numbers of idle gates.

        """
        circuits = []
        # Three loops, one for initial states, one for idle time, one for measurement basis
        for initial_state in self._bases:
            for steps in range(self.config["steps"]):
                for basis in self._bases:
                    circuit = QuantumCircuit(1)
                    if self.config["extra_zz_crosstalk"] != 0:
                        circuit = QuantumCircuit(2, 1)

                    # Initial states - prepare in |0>, |+> or |R>
                    self.prepare_initial_state(circuit, initial_state)

                    # Add idle gates
                    self.add_idle_gates(circuit, steps)

                    # Measurement basis
                    self.change_measurement_basis(circuit, basis)

                    if self.config["extra_zz_crosstalk"] != 0:
                        circuit.measure(0, 0)
                    else:
                        circuit.measure_all()

                    circuits.append(circuit)
        return circuits

    @staticmethod
    def fit_func(x, a, b, lambda_, omega):
        """Exponentially decaying oscillation fit function.

        This is used for calculating the oscillation frequency of qubit purity.

        Args:
            x (float or array-like): Independent variable (number of idle gates).
            a (float): Readout error fitting parameter.
            b (float): State preparation error fitting parameter.
            lambda_ (float): Decay rate fitting parameter.
            omega (float): Oscillation frequency fitting parameter.

        Returns:
            ndarray: fitting function datapoints.

        """
        return a + b * np.e ** (-1 * lambda_ * x) * np.cos(x * omega) ** 2

    def _analyze(self):
        """Analyze measurement results for the idle qubit oscillation frequency metric.

        Transforms raw counts into probabilities, computes the qubit purity for each initial
        state for each idle duration, then computes the purity oscillation frequency for
        each initial state, finally selects the largest frequency as the idle qubit oscillation
        frequency. Returns relevant information in a dictionary.

        Returns:
            dict: {
              "initial_state_0_purities": list,
              "initial_state_0_fit_result": {"popt": array, "pcov": array},
              "initial_state_0_oscillation_frequency": float,
              "initial_state_+_purities": list,
              "initial_state_+_fit_result": {"popt": array, "pcov": array},
              "initial_state_+_oscillation_frequency": float,
              "initial_state_R_purities": list,
              "initial_state_R_fit_result": {"popt": array, "pcov": array},
              "initial_state_R_oscillation_frequency": float,
              "idle_qubit_oscillation_frequency": float
            }

        """
        self.measurements_to_probabilities()

        index = 0
        result = {}
        fit_freqs = []

        for initial_state in self._bases:
            purities = []
            for _ in range(self.config["steps"]):
                purity = 1
                for _ in self._bases:
                    probs = self._experiment_data["meas_prob"][index]
                    prob_0 = probs["0"] if "0" in probs.keys() else 0
                    purity += (2 * prob_0 - 1) ** 2
                    index += 1
                purity /= 2
                purities.append(purity)
            self._purities.append(purities)
            result[
                f"initial_state_{self._initial_state_names[initial_state]}_purities"
            ] = purities

            try:
                popt, pcov = curve_fit(
                    self.fit_func,
                    self.config["idle_times"],
                    purities,
                    p0=[0.5, 0.5, 0.5, 0.5],
                )
            except RuntimeError:
                popt = [0, 0, 0, 0]
                pcov = None
            self._fit_results.append(popt)
            result[
                f"initial_state_{self._initial_state_names[initial_state]}_fit_result"
            ] = {"popt": popt, "pcov": pcov}

            # Discard if oscillation amplitude is too small
            fit_freq = np.absolute(popt[3])
            if np.absolute(popt[1]) < 0.01:
                fit_freq = 0
            fit_freqs.append(fit_freq)
            result[
                f"initial_state_{self._initial_state_names[initial_state]}_oscillation_frequency"
            ] = float(fit_freq)

        result["idle_qubit_oscillation_frequency"] = float(max(fit_freqs))
        return result

    def _plot(self, axes):
        """Plot the purity and the fitted exponentially decaying oscillation function for each initial state.

        Args:
            axes (matplotlib.axes.Axes): Axes to draw the plots on.

        Returns:
            matplotlib.legend.Legend: Legend for the plot.

        """
        colours = ["blue", "orange", "green"]
        for initial_state in [self._Z, self._X, self._Y]:
            axes.plot(
                self.config["idle_times"],
                self._purities[initial_state],
                linestyle="",
                c=colours[initial_state],
                marker="x",
                label=f"|{self._initial_state_names[initial_state]}> state purities",
            )
            axes.plot(
                self.config["idle_times"],
                self.fit_func(
                    self.config["idle_times"], *self._fit_results[initial_state]
                ),
                c=colours[initial_state],
                label=f"|{self._initial_state_names[initial_state]}> state fitted oscillation",
            )
            axes.set_xlabel(r"t $(n_{\mathrm{1q gates}})$")
            axes.set_ylabel("Purity")

        return axes.legend()
