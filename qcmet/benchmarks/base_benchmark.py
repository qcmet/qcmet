"""base_benchmark.py.

This module defines the API for all benchmarks in the QCMet framework. It provides
the BaseBenchmark abstract class, which establishes the structure and required
methods for implementing quantum computing benchmarks.

Features:
- Defines the BaseBenchmark abstract base class.
- Enforces implementation of _generate_circuits and _analyze methods.
- Supports optional plotting functionality.
- Handles benchmark registration via a registry mechanism.
- Manages experiment data, circuit serialization, and result persistence.
- Integrates with BaseDevice derived objects for benchmark execution.

All benchmark implementations in QCMet must inherit from BaseBenchmark and
adhere to its interface.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Counter, Dict, List, Optional
from uuid import uuid4

if TYPE_CHECKING:
    import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from qiskit import QuantumCircuit, qasm3, qpy

from qcmet.core.exceptions import MeasurementOutcomesExistError
from qcmet.core.file_manager import FileManager
from qcmet.devices.base_device import BaseDevice

_registry: dict[str, type] = {}
"""Global registry mapping class names to their class objects.

Keys:
    str: The name of the class.
Values:
    type: The class object associated with that name.
"""


def register_class(cls) -> None:
    """Register a class in the global registry for later lookup.

    This function adds the given class to a module-level dictionary `_registry`,
    using the class’s name as the key. Subsequent code can instantiate or inspect
    registered classes by name via `_registry`.

    Args:
        cls (type): The class to register.

    """
    _registry[cls.__name__] = cls


class BaseBenchmark(ABC):
    """Abstract base class for quantum computing benchmarks.

    This class defines the structure for benchmarks that generate quantum circuits, run them on
    quantum devices or simulators, analyze the results, and optionally plot
    them.

    Subclasses must implement:
        - _generate_circuits
        - _analyze

    Optionally, subclasses can implement:
        - _plot

    Attributes:
        name (str): Name of the benchmark.
        qubits (int | List[int]): List of qubit indices.
        config (dict): Configuration dictionary for benchmark-specific settings.
        save_enabled (bool): Whether saving is enabled.
        file_manager (FileManager | None): Manages file saving and loading.

    """

    def __init_subclass__(cls, **kwargs):
        """Add each benchmark to a registry of benchmarks."""
        super().__init_subclass__(**kwargs)
        register_class(cls)

    def __init__(
        self,
        name: str,
        qubits: int | List[int],
        save_path: Optional[str | Path | FileManager] = None,
    ):
        """Define constructor for benchmarks.

        Args:
            name (str): This is the name of the benchmark
            qubits (int | List[int]): The number of qubits as either a list of qubit
                indices or int specifying number of qubits. If int, qubit indices default
                to 0-indexed list.
            save_path (str | Path | FileManager, optional): Path to save benchmark
                outputs. Defaults to None.

        Raises:
            ValueError: If an invalid parameter is set, such as negative number of qubits

        """
        if isinstance(qubits, int):
            if qubits < 0:
                raise ValueError("Invalid number of qubits specified")
            else:
                self.qubits = np.arange(0, qubits).tolist()
        elif isinstance(qubits, list):
            if not all(isinstance(qb, int) for qb in qubits):
                raise ValueError("Qubits indices must be int")
            else:
                self.qubits = qubits

        # Benchmark config specific to the benchmark should be stored in the config dictionary

        self.config: Dict[str, Any] = {}
        self.name: str = name
        self._experiment_data: DataFrame = None

        self.save_enabled = True if save_path else False
        if isinstance(save_path, str):
            self.file_manager = FileManager(self.name, Path(save_path))
        elif isinstance(save_path, Path):
            self.file_manager = FileManager(self.name, save_path)
        elif isinstance(save_path, FileManager):
            self.file_manager = save_path
        else:
            self.file_manager = None

    @abstractmethod
    def _generate_circuits(self) -> List[QuantumCircuit] | Dict[str, Any]:
        """Generate circuits for benchmark.

        If the method returns a dictionary, it must contain the key 'circuit' and the value
        should be a QuantumCircuit object.

        Returns:
            List[QuantumCircuit] | Dict[str, any]: A list or dictionary of quantum circuits.

        """
        pass

    @abstractmethod
    def _analyze(self) -> Dict[str, Any]:
        """Analyze circuit measurements to calculate benchmark results.

        Circuit outcomes can be accessed using experiment_data.

        Returns:
            Dict[str, any]: A dictionary of results from the benchmark

        """
        pass

    def _plot(self, axes):
        """Plot benchmark results, optional.

        When implementing this function, all plotting should be done on
        an axes object passed into this function.
        """
        print("No plotting implemented for this benchmark")
        return None

    def has_plotting(self):
        """Check if  _plot function is implemented in benchmark.

        Returns:
            Bool: True if benchmark class has a plot function

        """
        return self.__class__._plot is not BaseBenchmark._plot

    def plot(self, axes: Optional[matplotlib.axes._axes.Axes] = None):
        """Plot benchmark result, user facing.

        Args:
            axes (matplotlib.axes._axes.Axes, optional): Plot will use axes if
                provided. Defaults to None.

        """
        if self.has_plotting():
            if axes is None:
                fig, ax = plt.subplots()
                self._plot(ax)
                return fig, ax
            else:
                self._plot(axes)
        else:
            print("No plotting implemented for this benchmark")
            return None

    @property
    def experiment_data(self):
        """Getter for experiment_data dataframe.

        Returns:
            pd.Dataframe : a dataframe where each row represents a circuit and all metadata represented to the circuit

        """
        if self._experiment_data is None:
            raise AttributeError("Experiment data not yet generated")
        else:
            return self._experiment_data

    @experiment_data.setter
    def experiment_data(self, circuits: List[QuantumCircuit] | Dict[str, Any]):
        """Setter for the experiment_data dataframe.

        When implementing a benchmark, as part of the generate_circuits method, the
        final list of circuits or list of dictionaries which contains the
        circuit and any metadata should be assigned to experiment_data. The
        setter then generates the correct dataframe structure.

        Args:
            circuits (List[QuantumCircuit] | Dict[str, any]): The list of
                circuit data generated by the benchmark.

        Raises:
            ValueError: If the circuits are not provided in the correct format,
                then assigning them to experiment_data will raise an error.

        """
        if not isinstance(circuits, list):
            raise ValueError("Circuits must be supplied in list")
        elif not circuits:  # empty list
            raise ValueError("No circuits supplied")
        elif all(isinstance(item, QuantumCircuit) for item in circuits):
            self._experiment_data = self._circs_to_df(
                [self._circ_with_metadata_dict(qc) for qc in circuits]
            )
        # Check if list of dicts containing QuantumCircuits
        elif all(isinstance(item, dict) for item in circuits):
            if all("circuit" in circ_dict for circ_dict in circuits):
                self._experiment_data = self._circs_to_df(circuits)
            else:
                raise ValueError("Circuits in wrong format")
        else:
            raise ValueError("Circuits in wrong format")

    @property
    def num_qubits(self) -> int:
        """Number of qubits in this benchmark.

        Always returns the current length of `self.qubits`, so it
        stays in sync if `self.qubits` is ever modified.
        """
        return len(self.qubits)

    @property
    def circuits(self) -> List[QuantumCircuit]:
        """Gets all benchmark circuits.

        Returns:
            List[QuantumCircuit]: list of all benchmark circuits.

        """
        if self._experiment_data is not None:
            return self._experiment_data["circuit"].to_list()
        else:
            raise AttributeError(f"Circuits not generated for {self.name}!")

    def _save_circuits(self, binary: bool = True):
        """Save quantum circuits.

        Args:
            binary (bool, optional): Choice to save circuits in binary or human
            readable format. Defaults to True.

        """
        if binary:
            with open(self.file_manager.get_data_path() / "circuits.qpy", "wb") as file:
                qpy.dump(self.circuits(), file)
        else:
            for i, circ in enumerate(self.circuits()):
                with open(
                    self.file_manager.get_data_path() / f"circuit_{i}.qasm", "w"
                ) as file:
                    qasm3.dump(circ, file)

    def run(self, device: BaseDevice = None, num_shots: int = 1024, **kwargs):
        """Run benchmark.

        If no device is provided, run_offline is executed which provides all necessary data to run benchmark offline in a saved
        directory.

        Args:
            device (BaseDevice, optional): Device to run benchmark on. Defaults to None.
            num_shots (int, optional): Number of measurements per circuit. Defaults to 1024.
            **kwargs (Dict[str, any]): Optional keyword arguments passed to device in _runtime_params.

        Raises:
            MeasurementOutcomesExistError: When benchmark already has measurements

        """
        self._runtime_params: Dict[str, Any] = {
            "num_shots": num_shots,
            "device": device,
        } | kwargs

        if "circuit_measurements" in self._experiment_data:
            raise MeasurementOutcomesExistError()
        else:
            if device is not None:
                return self._run_online()
            else:
                return self._run_offline()

    def generate_circuits(self) -> None:
        """Generate benchmark circuits, user facing.

        The benchmark circuits are routed based on user-defined qubit indices (if specified).

        """
        self.experiment_data = self._generate_circuits()
        if isinstance(self.qubits, list):
            routed_circuits = []
            qc_index = QuantumCircuit(max(self.qubits) + 1)
            for qc in self.experiment_data["circuit"]:
                qc_routed = qc_index.compose(qc, qubits=self.qubits)
                routed_circuits.append(qc_routed)
            self.experiment_data["circuit"] = routed_circuits
        else:
            pass
        if self.save_enabled:
            self.save()

    def _circ_with_metadata_dict(self, circ, **kwargs):
        row = {
            "hash": self._hash_circuit(circ),
            "id": str(uuid4()),
            "circuit": circ,
        }
        return row | kwargs

    def _circs_to_df(self, circs_with_metadata: List[Dict[str, Any]]):
        """Convert all circuits to experiment dataframe format.

        Args:
            circs_with_metadata (List[Dict[str, Any]]): List of dictionaries
                where each dict contains the circuit and other circuit related metadata.

        Returns:
            pd.DataFrame: experiment data dataframe

        """
        return DataFrame(circs_with_metadata)

    def _run_online(self) -> List[Dict[str, int]]:
        """Run benchmark on BaseDevice.

        If a device is specified, benchmark is run online and measurements
        are loaded to the experiment data.
        """
        counts = self._runtime_params["device"].run(
            self.circuits,
            num_shots=self._runtime_params["num_shots"],
        )
        self.load_circuit_measurements(counts)

    def _run_offline(self):
        """Run benchmark offline when no device specified.

        This function saves all data to 2 json files that the user can then
        use at a later date. This includes the benchmarking circuits in a qasm
        format.

        Raises:
            ValueError: If no save path is set

        """
        copy_df = self.experiment_data
        qasm_col = []
        for circ in self.circuits:
            qasm_col.append(qasm3.dumps(circ))
        copy_df["circuits"] = qasm_col
        if self.file_manager:
            self.file_manager.save_json(
                self._runtime_params, "runtime_parameters", "intermediate"
            )
            return copy_df.to_json(
                self.file_manager.get_intermediate_path() / "exp.json"
            )
        else:
            raise ValueError("Save path not specified. Use self.set_save_path()")

    def measurements_to_probabilities(self):
        """Convert raw measurement counts to normalized probabilities.

        Takes measurement results from quantum circuits stored in the
        `experiment_data["circuit_measurements"]` DataFrame column and computes
        the corresponding probability for each outcome. Probabilities are
        calculated by dividing each count by the total number of shots.

        The normalized results are stored in a new column `experiment_data["meas_prob"]`.

        Returns:
            None: The method modifies the `experiment_data` in place.

        """
        self.experiment_data["meas_prob"] = self.experiment_data[
            "circuit_measurements"
        ].apply(
            lambda x: {
                key: val / self._runtime_params["num_shots"] for key, val in x.items()
            }
        )

    def load_circuit_measurements(
        self,
        circuit_measurements: List[Dict[str, int] | Counter] | Dict[str, int] | Counter,
    ) -> None:
        """Load measurement counts into the experiment_data DataFrame.

        Supports two modes:
        - Multiple circuits: pass a list of dicts/Counters, one per row.
        - Single circuit: pass a single dict/Counter if the DataFrame has exactly one row.

        The method assigns the measurement results to the
        'circuit_measurements' column.

        Args:
            circuit_measurements: Either
            - A list of dict or Counter, length must equal the number of rows in experiment_data.
            - A single dict or Counter, only valid when experiment_data has exactly one row.

        Raises:
            ValueError: If the input type or length does not match the DataFrame shape.

        """
        num_rows = self._experiment_data.shape[0]

        # Single-circuit mode
        if isinstance(circuit_measurements, (dict, Counter)):
            if num_rows != 1:
                raise ValueError(
                    f"Must supply measurements for all {num_rows} circuits"
                )
            else:
                self._experiment_data["circuit_measurements"] = [circuit_measurements]

        # Multi-circuit mode
        elif isinstance(circuit_measurements, list):
            if len(circuit_measurements) != num_rows:
                raise ValueError(
                    f"Expected {num_rows} measurement entries, got {len(circuit_measurements)}"
                )
            self._experiment_data["circuit_measurements"] = circuit_measurements

        else:
            raise ValueError(
                "circuit_measurements must be a dict/Counter or list of dicts/Counters"
            )

    def analyze(self):
        """Analyze measurements to return benchmark results.

        Returns:
            Dict : results of benchmark in dictionary format

        """
        self.result = self._analyze()
        if self.save_enabled:
            self.save()
            self.file_manager.save_json(self.result, "result", "results")
            if self.has_plotting():
                fig = plt.figure()
                axes = fig.gca()
                self.file_manager.save_plot(self.plot(axes), "plot")
        return self.result

    def save(self):
        """Save benchmark current state.

        Raises:
            ValueError: If no save path is set

        """
        if self.save_enabled:
            self._experiment_data.to_pickle(
                self.file_manager.get_data_path() / "dataframe.pkl"
            )
            self.file_manager.save_json(
                self.config, filename="experiment_config", subfolder="config"
            )
        else:
            raise ValueError("Save path not specified. Use self.set_save_path()")

    def set_save_path(self, save_path: str | Path):
        """Set benchmark save path if not set in class constructor.

        Args:
            save_path (str | Path): path to create bencmark output save folder

        """
        self.save_path = Path(save_path)
        self.save_enabled = True

    def _hash_circuit(self, circuit) -> str:
        try:
            qasm_str = qasm3.dumps(circuit)
            return hashlib.md5(qasm_str.encode("utf-8")).hexdigest()
        except TypeError:
            # Fallback
            qc_ops = str([[(op.name, op.qubits, op.params) for op in circuit.data]])
            return hashlib.md5(qc_ops.encode("utf-8")).hexdigest()

    def __call__(
        self,
        device: BaseDevice,
        num_shots: int = 1024,
        axes: Optional[matplotlib.axes._axes.Axes] = None,
        **kwargs,
    ):
        """Benchmark instance to be called as a function.

        Performs a composite operation by sequentially executing generate_circuits(), run() and analyze().

        Args:
            device (BaseDevice): Device to run benchmark on.
            num_shots (int, optional): Number of measurements per circuit. Defaults to 1024.
            axes (matplotlib.axes._axes.Axes, optional): Plot will use axes if provided. Defaults to None.
            **kwargs (Dict[str, any]): Optional keyword arguments passed to device in _runtime_params.

        """
        self.generate_circuits()
        self.run(device=device, num_shots=num_shots, **kwargs)
        self.analyze()
        self.plot(axes=axes)
        return self.result
