"""Cycle Benchmarking for Composite Process Fidelity.

This module implements cycle benchmarking, a protocol for characterizing
the fidelity of a repeated gate layer (cycle) on a quantum processor.
Based on: https://doi.org/10.1038/s41467-019-13068-7
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from pathlib import Path

    from qcmet.core import FileManager
import itertools
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import IGate, XGate, YGate, ZGate
from qiskit.transpiler import PassManager
from scipy.optimize import curve_fit

from qcmet.benchmarks import BaseBenchmark
from qcmet.utils import PauliTwirl


class CycleBenchmarking(BaseBenchmark):
    """Benchmark class for estimating composite process fidelity using cycle benchmarking.

    Cycle benchmarking characterizes the average fidelity of a repeated layered
    operation (called G or the "cycle") by measuring diagonal elements of the
    Pauli transfer matrix (PTM) across different cycle repetition counts.

    The protocol:
    1. Prepare an eigenstate of a Pauli operator P
    2. Apply m repetitions of the cycle G (with Pauli twirling)
    3. Measure in the P basis
    4. Average over many Pauli operators and random sequences
    5. Extract composite process fidelity from PTM elements

    Attributes:
        config (dict): Configuration dictionary containing:
            - g_layer (QuantumCircuit): The gate layer/cycle to benchmark
            - repetitions_list (list): List of cycle repetition counts [m1, m2, ...]
            - num_random_sequences (int): Number of random twirled sequences per Pauli
            - full_pauli_subspace (bool): Whether to use full Pauli subspace
            - subspace_size (int): Size of Pauli subspace if not using full
            - fidelity_method (str): Method to calculate fidelity ('fit' or 'ratio')

    """

    # Pauli gates for creating eigenstates and measurement bases
    _pauli_gates: List = [XGate, YGate, ZGate, IGate]
    _pauli_labels: dict[Any, str] = {XGate: "X", YGate: "Y", ZGate: "Z", IGate: "I"}
    _pauli_to_mat: dict[str, np.ndarray] = {
        "X": XGate().to_matrix(),
        "Y": YGate().to_matrix(),
        "Z": ZGate().to_matrix(),
        "I": IGate().to_matrix(),
    }

    def __init__(
        self,
        g_layer: QuantumCircuit,
        repetitions_list: list,
        num_random_sequences: int = 10,
        full_pauli_subspace: bool = True,
        subspace_size: int | None = None,
        seed: int | np.random.Generator | None = None,
        fidelity_method: str = "ratio",
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the cycle benchmarking benchmark.

        Args:
            g_layer (QuantumCircuit): The repeated gate layer (cycle) to benchmark.
                Must be a QuantumCircuit on n qubits.
            repetitions_list (list): List of cycle repetition counts to test.
                Example: [2, 4, 8, 10]
            num_random_sequences (int, optional): Number of random Pauli-twirled
                sequences per Pauli channel. Defaults to 10.
            full_pauli_subspace (bool, optional): Whether to use the full Pauli
                subspace. For n qubits, this is 4^n - 1 operators (excluding all-I).
                Defaults to True.
            subspace_size (int, optional): Size of random Pauli subspace if
                full_pauli_subspace is False. Defaults to None.
            seed (int, optional): Seed for reproducibility. Used to  create
                a `np.random.Generator` object.
            fidelity_method (str, optional): Method to calculate composite process
                fidelity. Either 'fit' (exponential fit) or 'ratio' (direct ratio).
                Defaults to 'ratio'.
            save_path (str | Path | FileManager | None, optional): Directory path to save
                results. Defaults to None.

        Raises:
            ValueError: If fidelity_method is not 'fit' or 'ratio'.
            ValueError: If subspace_size is invalid when not using full subspace.

        """
        # Get number of qubits from g_layer
        num_qubits = g_layer.num_qubits

        super().__init__("CycleBenchmarking", qubits=num_qubits, save_path=save_path)

        if fidelity_method not in ["fit", "ratio"]:
            raise ValueError("fidelity_method must be either 'fit' or 'ratio'")

        self.config["g_layer"] = g_layer
        self.config["repetitions_list"] = repetitions_list
        self.config["num_random_sequences"] = num_random_sequences
        self.config["full_pauli_subspace"] = full_pauli_subspace
        self.config["subspace_size"] = subspace_size
        self.config["fidelity_method"] = fidelity_method
        self.config["seed"] = seed
        self._rng: np.random.Generator = np.random.default_rng(self.config["seed"])
        # Generate Pauli subspace
        if full_pauli_subspace:
            self.pauli_list = self._get_full_pauli_subspace()
            self.config["subspace_size"] = len(self.pauli_list)
        else:
            if subspace_size is None:
                raise ValueError(
                    "subspace_size must be specified when full_pauli_subspace=False"
                )
            if subspace_size > (4**num_qubits - 1):
                raise ValueError(
                    f"Subspace size {subspace_size} larger than full subspace {4**num_qubits - 1}"
                )
            self.pauli_list = self._get_random_pauli_subspace()

    def _get_full_pauli_subspace(self):
        """Generate all non-identity Pauli strings for n qubits.

        Returns:
            list: List of tuples of Pauli gate classes, excluding all-I.

        """
        p_list = list(itertools.product(self._pauli_gates, repeat=self.num_qubits))
        # Remove the all-identity element
        p_list.remove(tuple(IGate for _ in range(self.num_qubits)))
        return p_list

    def _get_random_pauli_subspace(self):
        """Generate a random subset of Pauli strings for n qubits.

        Returns:
            list: Random list of Pauli string tuples, excluding all-I.

        """
        subspace_size = self.config["subspace_size"]
        p_list = []

        # Generate random Paulis until we have the desired subspace size
        while len(p_list) < subspace_size:
            pauli_string = tuple(
                self._rng.choice(self._pauli_gates, size=self.num_qubits)
            )
            # Exclude all-identity
            if pauli_string != tuple(IGate for _ in range(self.num_qubits)):
                if pauli_string not in p_list:
                    p_list.append(pauli_string)

        return p_list

    def _get_pauli_label(self, pauli_channel):
        """Convert Pauli gate tuple to string label.

        Args:
            pauli_channel (tuple): Tuple of Pauli gate classes.

        Returns:
            str: String label like 'XY', 'IZ', etc.

        """
        return "".join([self._pauli_labels[gate] for gate in pauli_channel])

    def _apply_state_basis_change(self, qc, pauli_gate, qubit):
        """Apply basis rotation to prepare Pauli eigenstate.

        Args:
            qc (QuantumCircuit): Circuit to modify.
            pauli_gate (Gate class): Pauli gate class (XGate, YGate, ZGate, IGate).
            qubit (int): Qubit index to apply rotation to.

        """
        if pauli_gate is XGate:
            qc.h(qubit)
        elif pauli_gate is YGate:
            qc.h(qubit)
            qc.s(qubit)
        # For ZGate and IGate, no rotation needed

    def _apply_measurement_basis_change(self, qc, pauli_gate, qubit):
        """Apply basis rotation for Pauli measurement.

        Args:
            qc (QuantumCircuit): Circuit to modify.
            pauli_gate (Gate class): Pauli gate class (XGate, YGate, ZGate, IGate).
            qubit (int): Qubit index to apply rotation to.

        """
        if pauli_gate is XGate:
            qc.h(qubit)
        elif pauli_gate is YGate:
            qc.sdg(qubit)
            qc.h(qubit)
        # For ZGate and IGate, no rotation needed

    def _generate_twirled_cycles(self, repetitions):
        """Generate Pauli-twirled versions of the repeated cycle.

        Args:
            repetitions (int): Number of times to repeat the cycle.

        Returns:
            list: List of QuantumCircuits with Pauli twirling applied.

        """
        # Create the repeated cycle
        rep_cycle = self.config["g_layer"].repeat(repetitions)
        rep_cycle = rep_cycle.decompose()

        # Apply Pauli twirling using PassManager
        pm = PassManager([PauliTwirl(seed=self._rng)])

        twirled_circuits = []
        for _ in range(self.config["num_random_sequences"]):
            twirled_qc = pm.run(rep_cycle)
            twirled_circuits.append(twirled_qc)

        return twirled_circuits

    def _generate_circuits(self):
        """Generate cycle benchmarking circuits.

        For each repetition count m:
            For each Pauli channel P:
                For each random sequence:
                    1. Prepare |+⟩_P eigenstate
                    2. Apply m twirled cycles
                    3. Measure in P basis

        Returns:
            list: List of dictionaries containing circuits and metadata.

        """
        circuits = []

        for m in self.config["repetitions_list"]:
            # Get twirled cycles for this repetition count
            twirled_cycles = self._generate_twirled_cycles(m)

            for pauli_channel in self.pauli_list:
                pc_label = self._get_pauli_label(pauli_channel)

                for twirled_cycle in twirled_cycles:
                    qc = QuantumCircuit(self.num_qubits, self.num_qubits)

                    # Apply state basis changing operations (prepare Pauli eigenstate)
                    for i, pauli in enumerate(pauli_channel):
                        self._apply_state_basis_change(qc, pauli, i)
                        qc.barrier()

                    # Append the twirled cycle
                    qc = qc.compose(twirled_cycle)
                    qc.barrier()

                    # Apply measurement basis changing operations
                    for i, pauli in enumerate(pauli_channel):
                        self._apply_measurement_basis_change(qc, pauli, i)

                    # Measure all qubits
                    for i in self.qubits:
                        qc.measure(i, i)

                    circuits.append(
                        self._circ_with_metadata_dict(qc, m=m, pauli_channel=pc_label)
                    )

        return circuits

    def _calculate_pauli_expectation(self, pauli_string, counts, num_shots):
        """Calculate expectation value of Pauli operator from measurement counts.

        Args:
            pauli_string (str): Pauli string like 'XY', 'IZ', etc.
            counts (dict): Measurement counts dictionary.
            num_shots (int): Total number of shots.

        Returns:
            float: Expectation value of the Pauli operator.

        """
        expectation = 0
        for state, count in counts.items():
            # Calculate parity (-1)^(number of 1s in measured bits for non-I Paulis)
            parity = 1
            for bit, pauli in zip(state, pauli_string, strict=False):
                if pauli != "I":
                    if int(bit) == 1:
                        parity *= -1
            expectation += parity * count

        return expectation / num_shots

    def _get_ptm_elements(self):
        """Calculate PTM diagonal elements for each repetition count.

        Returns:
            dict: Nested dictionary {m: {pauli_string: ptm_value}}

        """
        ptm_elements = {}

        for m in self.config["repetitions_list"]:
            ptm_for_m = {}

            # Filter data for this repetition count
            m_data = self._experiment_data[self._experiment_data["m"] == m]

            for pauli_channel in self.pauli_list:
                pc_label = self._get_pauli_label(pauli_channel)

                # Filter for this Pauli channel
                pc_data = m_data[m_data["pauli_channel"] == pc_label]

                # Calculate average expectation over all random sequences
                total_exp = 0
                for _, row in pc_data.iterrows():
                    exp = self._calculate_pauli_expectation(
                        pc_label,
                        row["circuit_measurements"],
                        self._runtime_params["num_shots"],
                    )
                    total_exp += exp

                avg_exp = total_exp / self.config["num_random_sequences"]
                ptm_for_m[pc_label] = avg_exp

            ptm_elements[m] = ptm_for_m

        return ptm_elements

    def _get_cycle_fidelities(self, ptm_elements):
        """Calculate cycle fidelities from PTM elements.

        Fidelity for m cycles: F_m = (1 + sum of PTM diagonal elements) / d^2
        where d = 2^n is the dimension of the Hilbert space.
        If full Pauli subspace is not used, then d^2 is replaced with number of
        Pauli elements.

        Args:
            ptm_elements (dict): PTM elements from _get_ptm_elements.

        Returns:
            list: Cycle fidelities for each repetition count.

        """
        fidelities = []
        if self.config["full_pauli_subspace"]:
            d_squared = (2**self.num_qubits) ** 2
        else:
            d_squared = self.config["subspace_size"]

        for m in self.config["repetitions_list"]:
            ptm_sum = sum(ptm_elements[m].values())
            fidelity = (1 + ptm_sum) / d_squared
            fidelities.append(fidelity)

        return fidelities

    @staticmethod
    def fit_func(x, a, b, c):
        """Exponential decay fit function for fidelity vs cycle count.

        Args:
            x (array): Cycle repetition counts.
            a (float): Amplitude parameter.
            b (float): Decay rate parameter.
            c (float): Baseline offset.

        Returns:
            array: Fitted fidelity values.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return a * np.exp(-b * x) + c

    def _analyze(self):
        """Analyze measurement results to calculate composite process fidelity.

        Returns:
            dict: Results including PTM elements, cycle fidelities, and
                composite process fidelity.

        """
        # Calculate PTM elements
        ptm_elements = self._get_ptm_elements()

        # Calculate cycle fidelities
        cycle_fidelities = self._get_cycle_fidelities(ptm_elements)

        # Calculate composite process fidelity
        if self.config["fidelity_method"] == "fit":
            result = self._analyze_with_fit(cycle_fidelities)
        else:  # ratio
            result = self._analyze_with_ratio(ptm_elements)

        # Store intermediate results
        result["ptm_elements"] = ptm_elements
        result["cycle_fidelities"] = cycle_fidelities

        return result

    def _analyze_with_fit(self, cycle_fidelities):
        """Calculate composite process fidelity using exponential fit.

        Args:
            cycle_fidelities (list): Cycle fidelities for each repetition count.

        Returns:
            dict: Results including fit parameters and composite process fidelity.

        """
        # Add soft constraint that fidelity=1 when m=0
        repetitions_with_zero = np.append([0], self.config["repetitions_list"])
        fidelities_with_one = np.append([1.0], cycle_fidelities)

        try:
            popt, pcov = curve_fit(
                self.fit_func,
                repetitions_with_zero,
                fidelities_with_one,
                p0=[1.0, 0.01, 0.0],
                maxfev=10000,
            )

            # Composite process fidelity is F(1) from the fit
            composite_fidelity = self.fit_func(1, *popt)

            return {
                "composite_process_fidelity": float(composite_fidelity),
                "method": "fit",
                "fit_params": {"popt": popt.tolist(), "pcov": pcov.tolist()},
                "success": True,
            }

        except Exception as e:
            print(f"Failed to fit cycle fidelity data: {e}")
            return {
                "composite_process_fidelity": None,
                "method": "fit",
                "success": False,
                "error": str(e),
            }

    def _analyze_with_ratio(self, ptm_elements):
        """Calculate composite process fidelity using ratio method.

        This method uses the formula:
        F = (1/K+1) * sum_P [ (lambda_P(m_j) / lambda_P(m_i))^(1/(m_j - m_i)) ]

        Args:
            ptm_elements (dict): PTM elements from _get_ptm_elements.

        Returns:
            dict: Results including composite process fidelity for each depth pair.

        """
        num_depths = len(self.config["repetitions_list"])

        # Calculate fidelities for all pairs of depths
        fidelity_pairs = {}

        for i in range(num_depths):
            for j in range(i + 1, num_depths):
                m_i = self.config["repetitions_list"][i]
                m_j = self.config["repetitions_list"][j]

                sum_ratios = 1  # Start with 1 for the identity term

                # Sum over all Pauli channels
                for pauli_label in ptm_elements[m_i].keys():
                    lambda_i = ptm_elements[m_i][pauli_label]
                    lambda_j = ptm_elements[m_j][pauli_label]

                    # Avoid division by zero or negative ratios
                    if lambda_i != 0:
                        ratio = lambda_j / lambda_i
                        if ratio > 0:
                            sum_ratios += ratio ** (1 / (m_j - m_i))

                # Calculate composite fidelity for this pair
                composite_fidelity = sum_ratios / (self.config["subspace_size"] + 1)
                pair_label = f"m={m_i}_to_m={m_j}"
                fidelity_pairs[pair_label] = float(composite_fidelity)

        return {
            "composite_process_fidelity": fidelity_pairs,
            "method": "ratio",
            "success": True,
        }

    def _plot(self, axes):
        """Plot cycle fidelity vs number of cycles.

        Args:
            axes (matplotlib.axes.Axes): Matplotlib axes object to draw the plot on.

        """
        if "cycle_fidelities" not in self.result:
            print("No cycle fidelity data to plot")
            return

        cycle_fidelities = self.result["cycle_fidelities"]

        axes.scatter(
            self.config["repetitions_list"],
            cycle_fidelities,
            color="black",
            marker="x",
            label="Measured fidelity",
        )

        # If using fit method, plot the fitted curve
        if (
            self.config["fidelity_method"] == "fit"
            and "fit_params" in self.result
            and self.result["success"]
        ):
            popt = self.result["fit_params"]["popt"]
            x_fit = np.linspace(0, max(self.config["repetitions_list"]), 100)
            y_fit = self.fit_func(x_fit, *popt)

            axes.plot(
                x_fit,
                y_fit,
                color="black",
                ls="--",
                label=f"Fit: F(1)={self.result['composite_process_fidelity']:.4f}",
            )

        axes.set_xlabel("Number of cycles (m)")
        axes.set_ylabel("Cycle Fidelity")
        axes.set_title("Cycle Benchmarking")
        axes.set_xlim(0, max(self.config["repetitions_list"]) * 1.1)
        axes.set_ylim(0, 1.1)
        axes.legend()
        axes.grid(True, alpha=0.3)
