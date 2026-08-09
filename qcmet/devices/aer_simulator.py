"""AerSimulator device backend for quantum circuit benchmarking.

This module provides a concrete implementation of the `QiskitDevice` interface,
wrapping Qiskit's `AerSimulator` as a virtual backend for running quantum circuits.
It supports noise modeling, optimization via Qiskit's transpiler, and conversion
of measurement results into big-endian bitstring format.

Classes:
    AerSimulator_Base - a simulated backend device for executing and analyzing quantum circuits.
"""

import numpy as np
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

    def __init__(self, noise_model: NoiseModel = None, basis_gates=None, exact_probabilities:bool=False, **kwargs,):
        """Initialize the Aer simulator device.

        Args:
            noise_model (NoiseModel, optional): A Qiskit Aer noise model to simulate noisy execution.
            basis_gates (list[str], optional): Restrict transpiled circuits to specific gate set.
            exact_probabilities (bool): Determines whether finite shot error is included in simulations. 
                Defaults to False.
            **kwargs: Additional keyword arguments passed through to the AerSimulator constructor

        """
        super().__init__("aer_simulator")
        self.properties["noise_model"] = noise_model
        self.properties["exact_probabilities"] = exact_probabilities

        self.sim = AerSimulator(
            noise_model=noise_model, basis_gates=basis_gates, **kwargs
        )
        self.passmanager = generate_preset_pass_manager(
            optimization_level=0, backend=self.sim
        )

    def _run(self, circuits, num_shots):
        """Execute quantum circuits on the simulator with transpilation.

        By default, circuits are executed with finite shots and returned as reversed
        bitstring counts. If exact_probabilities=True, final measurements are removed,
        the final density matrix is saved, and exact computational-basis probabilities
        are returned scaled by num_shots.

        Args:
            circuits (QuantumCircuit | list[QuantumCircuit]):
                Qiskit quantum circuit or list of circuits to simulate.
            num_shots (int | float):
                Number of shots for sampled execution. When exact_probabilities=True,
                probabilities are multiplied by this value.
            exact_probabilities (bool):
                If False, return finite-shot sampled counts. If True, return exact
                probabilities scaled by num_shots, i.e. expected counts.

        Returns:
            dict | list[dict]:
                Measurement counts dictionary with bitstrings reversed, either for a
                single circuit or a list of circuits. In exact mode, the returned
                values are exact expected counts rather than sampled integer counts.

        """
        single_input = not isinstance(circuits, list)
        circuits_list = [circuits] if single_input else circuits

        if self.properties['exact_probabilities']:
            results = self._run_exact_expected_counts(circuits_list, num_shots)
        else:
            results = self._run_sampled_counts(circuits, num_shots)

        if single_input and isinstance(results, list):
            return results[0]

        return results

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

    def _run_sampled_counts(self, circuits, num_shots):
        """Run circuits using finite shots and return reversed bitstring counts.

        Args:
            circuits (QuantumCircuit | list[QuantumCircuit]):
                Qiskit quantum circuit or list of circuits to simulate.
            num_shots (int | float):
                Number of shots.

        Returns:
            dict | list[dict]:
                Sampled measurement counts with reversed bitstrings.

        """
        t_circuits = self.passmanager.run(circuits)
        results = self.sim.run(t_circuits, shots=int(num_shots)).result()

        counts = results.get_counts()

        if isinstance(counts, list):
            return [self.reverse_bitstrings(c) for c in counts]

        return self.reverse_bitstrings(counts)

    def _run_exact_expected_counts(self, circuits, num_shots):
        """Run circuits exactly and return probabilities scaled by num_shots.

        This removes final measurements, saves the final density matrix, extracts
        exact computational-basis probabilities, and scales them by num_shots to
        produce expected counts without finite-shot noise.

        Args:
            circuits (list[QuantumCircuit]):
                List of Qiskit quantum circuits to simulate.
            num_shots (int | float):
                Scaling factor for the exact probabilities.

        Returns:
            list[dict]:
                Expected counts with reversed bitstrings for each circuit.

        """
        circuits_with_save = []

        for qc in circuits:
            qc_dm = qc.copy()
            qc_dm = qc_dm.remove_final_measurements(inplace=False)
            qc_dm.save_density_matrix()
            circuits_with_save.append(qc_dm)

        t_circuits = self.passmanager.run(circuits_with_save)
        results = self.sim.run(t_circuits).result()

        scale = float(num_shots)

        out = []

        for i in range(len(t_circuits)):
            data = results.data(i)

            if "density_matrix" not in data:
                raise KeyError(
                    "'density_matrix' not found in result data. "
                    f"Available keys: {list(data.keys())}"
                )

            rho = data["density_matrix"]
            out.append(self._density_matrix_to_expected_counts(rho, scale))

        return out

    def _density_matrix_to_expected_counts(self, rho, scale):
        """Convert a density matrix to exact expected computational-basis counts.

        Args:
            rho:
                Qiskit density matrix result object.
            scale (float):
                Factor by which probabilities are multiplied, usually num_shots.

        Returns:
            dict:
                Expected counts with reversed bitstrings.

        """
        diag = np.real(np.diag(rho.data))
        diag = np.where(np.abs(diag) < 1e-15, 0.0, diag)

        n = rho.num_qubits

        return {
            format(j, f"0{n}b")[::-1]: float(p) * scale
            for j, p in enumerate(diag)
            if p > 0.0
        }
