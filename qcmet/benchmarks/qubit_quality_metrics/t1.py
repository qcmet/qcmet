"""T1 metric benchmark.

This module implements the T1 benchmark. Due to the differing levels of
access to quantum hardware, this benchmark allows for specifying the delay times
in terms of time or in terms of number of idle gates.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from qcmet.core import FileManager
import warnings

import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit

from qcmet.benchmarks import BaseBenchmark


class T1(BaseBenchmark):
    """Benchmark class for estimating the T1 relaxation time of a qubit using idle gate sequences.

    Attributes:
        config (dict): Configuration dictionary containing the number of idle gates per circuit.
        fit_result (dict): Stores the results of the exponential fit.
        success (bool): Indicates whether the curve fitting was successful.

    """

    def __init__(
        self,
        qubit_index: int = 0,
        num_idle_gates_per_circ: npt.ArrayLike | None = None,
        delay: npt.ArrayLike | None = None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the T1 benchmark.

        The benchmark can be specified with a range of idle gates or range of delay gates.
        The benchmark defaults to num_idle_gates for specifying the delay times.

        Args:
            qubit_index (int, optional): The qubit index for routing. Defaults to 0.
            num_idle_gates_per_circ (npt.ArrayLike, optional): Sequence of idle gate counts to use in circuits.
                Defaults to np.arange(1, 5000, 500).
            delay (npt.ArrayLike, optional): Sequence of delay gates to use in circuits. This is for alternate method
                to the idle_gates approach. Do not set an argument for num_idle_gates_per_circ for this approach.
                Units are in microseconds.
            save_path (str | Path | FileManager | None, optional): Directory path to save results. Defaults to None.

        Raises:
            ValueError: If num_idle_gates_per_circ and delay has been specified.

        """
        super().__init__("T1", qubits=[qubit_index], save_path=save_path)

        self.delay = False

        if num_idle_gates_per_circ is None and delay is None:
            self.config["num_idle_gates_per_circ"] = np.arange(1, 5000, 500)
        elif num_idle_gates_per_circ is not None:
            self.config["num_idle_gates_per_circ"] = num_idle_gates_per_circ
        elif delay is not None:
            self.delay = True
            self.config["delay"] = delay
        if num_idle_gates_per_circ is not None and delay is not None:
            raise ValueError("only specify num_idle_gates_per_circ or delay.")

    def _generate_circuits(self):
        """Generate quantum circuits for T1 measurement.

        Idle gates: Each circuit applies an X gate followed by a sequence of idle gates (
        identity gates) and a final measurement.

        Delay gates: Each circuit applies an X gate followed by a delay gate and a final measurement.

        Returns:
            list: A list of QuantumCircuit objects with varying numbers of idle gates or varying length of delay gates.

        """
        circuits = []
        if self.delay is False:
            for i in self.config["num_idle_gates_per_circ"]:
                qc = QuantumCircuit(1)
                qc.x(0)
                for _ in range(i):
                    qc.barrier()
                    qc.id(0)
                qc.barrier()
                qc.measure_all()
                circuits.append(qc)

        elif self.delay is True:
            for i in self.config["delay"]:
                qc = QuantumCircuit(1)
                qc.x(0)
                qc.barrier()
                qc.delay(float(i), unit="us")
                qc.barrier()
                qc.measure_all()
                circuits.append(qc)

        return circuits

    @staticmethod
    def exp_func(x, amp, dr):
        """Exponential decay function used to fit the T1 relaxation curve.

        Args:
            x (float or array-like): Independent variable (number of idle gates or delay duration).
            amp (float): Amplitude of the exponential decay.
            dr (float): Decay rate parameter, proportional to T1.

        Returns:
            float or array-like: Computed exponential decay values.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return amp * np.exp(-1 * x / dr)

    def _analyze(self):
        """Analyze the measurement results to estimate the T1 time.

        T1 time is estimated by fitting an exponential decay to the probability of measuring the excited state.

        Returns:
            dict: Dictionary containing the success flag, estimated T1 time, and fit parameters.

        """
        if not self.delay:
            self.x_data = self.config["num_idle_gates_per_circ"]
            guess = [1, 1000]
            t1_result_label = "T1 (t/t_[1q_gate])"

        if self.delay:
            self.x_data = self.config["delay"]
            guess = [1, 25]
            t1_result_label = "T1 (\u00b5s)"

        self.measurements_to_probabilities()
        self._experiment_data["p_1"] = self.experiment_data["meas_prob"].apply(
            lambda x: x.get("1")
        )
        try:
            popt, pcov = curve_fit(
                self.exp_func,
                self.x_data,
                self.experiment_data["p_1"],
                p0=guess,
                method="trf",
            )
            self.success = True
        except Exception as e:
            self.success = False
            print("Failed to fit T1 data")
            raise e

        self.fit_result = {"fit": {"popt": popt, "pcov": pcov}}

        result = {
            "success": self.success,
            t1_result_label: popt[1],
        } | self.fit_result

        return result

    def _plot(self, axes):
        """Plot measured p_1 vs num_idle_gates/delay and the fitted curve.

        Args:
            axes (matplotlib.axes.Axes): Matplotlib axes object to draw the plot on.

        """
        if not self.delay:
            x_label = r"t $(n_{\mathrm{1q gates}})$"

        if self.delay:
            x_label = r"Delay ($\mu$s)"

        x_to_fit = np.linspace(0, self.x_data[-1], 5000)
        fitted_probs = self.exp_func(x_to_fit, *self.fit_result["fit"]["popt"])

        axes.set_xlim((0, x_to_fit[-1]))
        axes.set_ylim((0, 1))
        axes.set_xlabel(x_label)
        axes.set_ylabel(r"$p_1$")
        axes.scatter(
            self.x_data,
            self.experiment_data["p_1"],
            color="black",
            marker="x",
            label="Measurements",
        )
        axes.plot(
            x_to_fit, fitted_probs, color="black", ls="--", label="Fitted equation"
        )
        axes.legend()
