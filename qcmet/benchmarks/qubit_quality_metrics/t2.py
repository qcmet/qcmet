"""T2 metric benchmark.

This module implements the T2 benchmark for both T2 Ramsey and T2 Hahn. Due to
differing levels of access to quantum hardware, this benchmark allows for 
specifying the delay times in terms of time or in terms of number of idle gates.

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


class T2(BaseBenchmark):
    """Benchmark class for estimating the T2 coherence time of a qubit.

    This benchmark supports two methods for measuring T2:
    - 'ramsey': Ramsey interference for measuring T2* (dephasing time)
    - 'hahn': Hahn echo (spin echo) for measuring T2 (coherence time)

    The Ramsey method is sensitive to low-frequency noise and detuning,
    while the Hahn echo method refocuses static field inhomogeneities.

    Attributes:
        config (dict): Configuration dictionary containing:
            - method (str): Either 'ramsey' or 'hahn'
            - num_idle_gates_per_circ (array-like): Number of idle gates per circuit
            - detuning_phase (float): Phase accumulation per idle gate for Ramsey 
                (defaults: np.pi/100 for idle_gates and np.pi/10 for delay gates)
            - delay (array-like): Length of delay per circuit
        fit_result (dict): Stores the results of the fit.
        success (bool): Indicates whether the curve fitting was successful.

    """

    def __init__(
        self,
        qubit_index: int = 0,
        method: str = "hahn",
        num_idle_gates_per_circ: npt.ArrayLike | None = None,
        detuning_phase: float | None = None,
        delay: npt.ArrayLike | None = None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the T2 benchmark.

        Args:
            qubit_index (int, optional): The qubit index for routing. Defaults to 0.
            method (str, optional): Method to use for T2 measurement. Either 'ramsey' or 'hahn'.
                Defaults to 'hahn'.
            num_idle_gates_per_circ (array-like, optional): Sequence of idle gate counts to use.
                For Ramsey: defaults to np.arange(1, 1000, 50)
                For Hahn: defaults to np.arange(1, 2000, 100)
            detuning_phase (float, optional): Phase per idle/delay gate to simulate 
                detuning in Ramsey.Defaults to np.pi/100 for idle gates approach and 
                np.pi/10 for delay gates approach. These defaults had produced 
                sensible fits for T2 and T2* with a noisy simulator.
            delay (array-like, optional): Sequence of delay gates to use in circuits. 
                This is for alternate methodto the idle_gates approach. Do not set
                an argument for num_idle_gates_per_circ when defining delay. Units
                are in microseconds.
            save_path (str | Path | FileManager | None, optional): Directory path to 
                save results. Defaults to None.

        Raises:
            ValueError: If method is not 'ramsey' or 'hahn'.
            ValueError: If num_idle_gates_per_circ and delay are both specified.

        """
        super().__init__("T2", qubits=[qubit_index], save_path=save_path)

        if method not in ["ramsey", "hahn"]:
            raise ValueError("method must be either 'ramsey' or 'hahn'")

        if num_idle_gates_per_circ is not None and delay is not None:
            raise ValueError("only specify num_idle_gates_per_circ or delay")

        self.config["method"] = method

        if delay is None:
            self.delay = False
            if num_idle_gates_per_circ is None:
                # Set default idle gate ranges based on method
                if method == "ramsey":
                    self.config["num_idle_gates_per_circ"] = np.arange(1, 1000, 50)
                else:  # hahn
                    self.config["num_idle_gates_per_circ"] = np.arange(1, 2000, 100)
            else:
                self.config["num_idle_gates_per_circ"] = num_idle_gates_per_circ
        else:
            self.delay = True
            self.config["delay"] = delay

        # Set default detuning_phase.
        if detuning_phase is None:
            if self.delay:
                detuning_phase = np.pi / 10
            else:
                detuning_phase = np.pi / 100

        self.config["detuning_phase"] = detuning_phase

    def _generate_circuits(self):
        """Generate quantum circuits for T2 measurement.

        Returns:
            list: A list of QuantumCircuit objects based on the selected method.

        """
        if self.config["method"] == "ramsey":
            return self._generate_ramsey_circuits()
        else:
            return self._generate_hahn_circuits()

    def _generate_ramsey_circuits(self):
        """Generate Ramsey interference circuits for T2* measurement.

        Circuit structure (idle_gates): SX - [ID + RZ(detuning)]*n - SX - Measure
        Circuit structure (delay): SX - delay - RZ(detuning) - SX - Measure

        Returns:
            list: A list of QuantumCircuit objects for Ramsey experiment.

        """
        circuits = []
        if self.delay is False:
            for num_gates in self.config["num_idle_gates_per_circ"]:
                qc = QuantumCircuit(1)
                qc.sx(0)  # First π/2 pulse
                for _ in range(int(num_gates)):
                    qc.id(0)
                    qc.barrier()
                    qc.rz(self.config["detuning_phase"], 0)  # Simulating detuned qubit
                qc.sx(0)  # Second π/2 pulse
                qc.measure_all()
                circuits.append(qc)

        if self.delay is True:
            for delay in self.config["delay"]:
                self.rotation_angle = self.config["detuning_phase"] * delay
                qc = QuantumCircuit(1)
                qc.sx(0)  # First π/2 pulse
                qc.delay(float(delay), unit="us")
                qc.barrier()
                qc.rz(self.rotation_angle, 0)  # Simulating detuned qubit
                qc.sx(0)  # Second π/2 pulse
                qc.measure_all()
                circuits.append(qc)

        return circuits

    def _generate_hahn_circuits(self):
        """Generate Hahn echo (spin echo) circuits for T2 measurement.

        Circuit structure: SX - [ID + RZ]^(n/2) - SX - SX - [ID + RZ]^(n/2) - SX - Measure

        The two SX gates in the middle form a π pulse (X gate) that refocuses dephasing.

        Returns:
            list: A list of QuantumCircuit objects for Hahn echo experiment.

        """
        circuits = []
        if self.delay is False:
            for num_gates in self.config["num_idle_gates_per_circ"]:
                qc = QuantumCircuit(1)
                qc.sx(0)  # First π/2 pulse

                # First half of evolution
                for _ in range(int(num_gates / 2)):
                    qc.id(0)
                    qc.rz(self.config["detuning_phase"], 0)

                # π pulse (refocusing pulse)
                qc.sx(0)
                qc.sx(0)

                # Second half of evolution
                for _ in range(int(num_gates / 2)):
                    qc.id(0)
                    qc.rz(self.config["detuning_phase"], 0)

                qc.sx(0)  # Final π/2 pulse
                qc.measure_all()
                circuits.append(qc)

        if self.delay is True:
            for delay in self.config["delay"]:
                self.rotation_angle = self.config["detuning_phase"] * delay
                qc = QuantumCircuit(1)
                qc.sx(0)  # First π/2 pulse

                # First half of evolution
                qc.delay(float(delay / 2), unit="us")
                qc.rz(self.rotation_angle, 0)

                # π pulse (refocusing pulse)
                qc.sx(0)
                qc.sx(0)

                # Second half of evolution
                qc.delay(float(delay / 2), unit="us")
                qc.rz(self.rotation_angle, 0)

                qc.sx(0)  # Final π/2 pulse
                qc.measure_all()
                circuits.append(qc)

        return circuits

    @staticmethod
    def ramsey_fit_func(x, amp, dr, f, phi, b):
        """Damped oscillation function for Ramsey interference.

        Models: amp * exp(-x/T2*) * cos(2πfx + φ) + b

        Args:
            x (float or array-like): Independent variable (number of idle gates or delay).
            amp (float): Amplitude of oscillation.
            dr (float): Decay rate (T2* time).
            f (float): Oscillation frequency.
            phi (float): Phase offset.
            b (float): Baseline offset.

        Returns:
            float or array-like: Computed damped oscillation values.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return amp * np.exp(x * -1 / dr) * np.cos(2 * np.pi * f * x + phi) + b

    @staticmethod
    def hahn_fit_func(x, amp, dr, b):
        """Exponential decay function for Hahn echo.

        Models: amp * exp(-x/T2) + b

        Args:
            x (float or array-like): Independent variable (number of idle gates or delay).
            amp (float): Amplitude of the exponential decay.
            dr (float): Decay rate (T2 time).
            b (float): Baseline offset.

        Returns:
            float or array-like: Computed exponential decay values.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return amp * np.exp(x * -1 / dr) + b

    def _analyze(self):
        """Analyze the measurement results to estimate the T2 time.

        For Ramsey: Fits to damped cosine and extracts T2*
        For Hahn: Fits to exponential decay and extracts T2

        Returns:
            dict: Dictionary containing the success flag, estimated T2 time, and fit parameters.

        """
        self.measurements_to_probabilities()

        if self.delay:
            self.x_data = self.config["delay"]
            self.x_units = " (\u00b5s)"
        else:
            self.x_data = self.config["num_idle_gates_per_circ"]
            self.x_units = " (t/t_[1q_gate])"

        if self.config["method"] == "ramsey":
            return self._analyze_ramsey()
        else:
            return self._analyze_hahn()

    def _analyze_ramsey(self):
        """Analyze Ramsey experiment results.

        Returns:
            dict: Results including T2* time and fit parameters.

        """
        # Extract probability of measuring |1⟩
        self._experiment_data["p_1"] = self.experiment_data["meas_prob"].apply(
            lambda x: x.get("1", 0)
        )

        try:
            # Frequency estimate based on detuning_phase
            freq_estimate = self.config["detuning_phase"] / (2 * np.pi)
            # Initial parameters: [amp, decay_rate, frequency, phase, baseline]
            p0 = [0.5, 1400, freq_estimate, 0, 0.5]
            popt, pcov = curve_fit(
                self.ramsey_fit_func,
                self.x_data,
                self.experiment_data["p_1"],
                p0=p0,
                method="trf",
                maxfev=50000,
            )
            self.success = True
        except Exception as e:
            self.success = False
            print(f"Failed to fit Ramsey data: {e}")
            raise e

        self.fit_result = {"fit": {"popt": popt, "pcov": pcov}}

        result = {
            "success": self.success,
            "T2*" + self.x_units: popt[1],
            "method": "ramsey",
        } | self.fit_result

        return result

    def _analyze_hahn(self):
        """Analyze Hahn echo experiment results.

        Returns:
            dict: Results including T2 time and fit parameters.

        """
        # Extract probability of measuring |0⟩ for Hahn echo
        self._experiment_data["p_0"] = self.experiment_data["meas_prob"].apply(
            lambda x: x.get("0", 0)
        )

        try:
            # Initial parameters: [amp, decay_rate, baseline]
            p0 = [0.5, 1400, 0.5]

            popt, pcov = curve_fit(
                self.hahn_fit_func,
                self.x_data,
                self.experiment_data["p_0"],
                p0=p0,
                method="trf",
                maxfev=5000,
            )
            self.success = True
        except Exception as e:
            self.success = False
            print(f"Failed to fit Hahn echo data: {e}")
            raise e

        self.fit_result = {"fit": {"popt": popt, "pcov": pcov}}

        result = {
            "success": self.success,
            "T2" + self.x_units: popt[1],
            "method": "hahn",
        } | self.fit_result

        return result

    def _plot(self, axes):
        """Plot measured probabilities vs num_idle_gates and the fitted curve.

        Args:
            axes (matplotlib.axes.Axes): Matplotlib axes object to draw the plot on.

        """
        if not self.delay:
            xlabel = r"t $(n_{\mathrm{1q gates}})$"
        else:
            xlabel = r"Delay ($\mu$s)"
        x_to_fit = np.linspace(0, self.x_data[-1], 5000)

        if self.config["method"] == "ramsey":
            fitted_probs = self.ramsey_fit_func(
                x_to_fit, *self.fit_result["fit"]["popt"]
            )
            y_data = self.experiment_data["p_1"]
            ylabel = r"$p_1^{R}$"
            title_suffix = "Ramsey (T2*)"
        else:
            fitted_probs = self.hahn_fit_func(x_to_fit, *self.fit_result["fit"]["popt"])
            y_data = self.experiment_data["p_0"]
            ylabel = r"$p_0^{E}$"
            title_suffix = "Hahn Echo (T2)"

        axes.set_xlim((0, x_to_fit[-1]))
        axes.set_ylim((0, 1))
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.scatter(
            self.x_data,
            y_data,
            color="black",
            marker="x",
            label="Measurement results",
        )
        axes.plot(x_to_fit, fitted_probs, color="black", ls="--", label="Fitted curve")
        axes.set_title(f"T2 Measurement - {title_suffix}")
        axes.legend()
