"""AerSimulator device backend for quantum circuit benchmarking.

This module provides a concrete implementation of the `QiskitDevice` interface,
wrapping Qiskit's `AerSimulator` as a virtual backend for running quantum circuits.
It supports noise modeling, optimization via Qiskit's transpiler, and conversion
of measurement results into big-endian bitstring format.

Classes:
    AerSimulator_Base - a simulated backend device for executing and analyzing quantum circuits.
"""

from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qcmet.devices.qiskit_device import QiskitDevice


class AerSimulatorBase(QiskitDevice):
    """Qiskit AerSimulator device implementation for quantum benchmarks.

    This class wraps the AerSimulator backend and applies an optimization pass
    before running circuits. It supports optional noise modeling and bitstring
    reversal to match big-endian interpretation of measurement results.

    Attributes:
        properties (dict): Contains optional 'noise_model' entry if provided.
        sim (AerSimulator): The simulator instance used for circuit execution.
        passmanager (PassManager): Qiskit's preset pass manager used to optimize circuits.

    """

    def __init__(self, noise_model: NoiseModel = None, basis_gates=None, **kwargs):
        """Initialize the Aer simulator device.

        Args:
            noise_model (NoiseModel, optional): A Qiskit Aer noise model to simulate noisy execution.
            basis_gates (list[str], optional): Restrict transpiled circuits to specific gate set.
            **kwargs: Additional keyword arguments passed through to the AerSimulator constructor

        """
        super().__init__("aer_simulator")
        self.properties["noise_model"] = noise_model
        self.sim = AerSimulator(
            noise_model=noise_model, basis_gates=basis_gates, **kwargs
        )
        self.passmanager = generate_preset_pass_manager(
            optimization_level=0, backend=self.sim
        )

    def _run(self, circuits, num_shots):
        """Execute quantum circuits on the simulator with transpilation.

        Applies the preset optimization pass manager to all input circuits,
        runs them on the AerSimulator, and returns reversed (big-endian) bitstring counts.

        Args:
            circuits (list[QuantumCircuit]): List of Qiskit quantum circuits to simulate.
            num_shots (int): Number of shots (repetitions) for each circuit execution.

        Returns:
            dict | list[dict]: Measurement counts dictionary with bitstrings reversed,
            either for a single circuits or a list of circuits.

        """
        t_circuits = self.passmanager.run(circuits)
        results = self.sim.run(t_circuits, shots=num_shots).result()
        counts = results.get_counts()
        if isinstance(counts, list):
            return [self.reverse_bitstrings(c) for c in counts]
        else:
            return self.reverse_bitstrings(counts)

    @staticmethod
    def reverse_bitstrings(counts_dict):
        """Reverse bitstrings in a measurement count dictionary.

        This converts between little and big endian.

        Args:
            counts_dict (dict): Original measurement counts with bitstrings as keys.

        Returns:
            dict: A new dictionary with bitstrings reversed.

        """
        big_endian_counts = {}
        for key, val in counts_dict.items():
            big_endian_counts[key[::-1]] = val
        return big_endian_counts

    def get_properties(self):
        """Return device metadata including noise model.

        Returns:
            dict: Dictionary containing device properties.

        """
        return self.properties
