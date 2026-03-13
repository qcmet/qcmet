"""sequential_benchmark.py.

This module provides a utility class for executing a benchmark multiple times with
different numbers of qubits, and optionally with a user-defined fail condition
such that future runs will be aborted if the fail condition is met by a run.
"""
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional

from qiskit import QuantumCircuit

from qcmet.benchmarks.base_benchmark import BaseBenchmark
from qcmet.core import FileManager
from qcmet.devices.base_device import BaseDevice


class SequentialBenchmark(BaseBenchmark, ABC):
    """Abstract utility class for running a BaseBenchmark with different numbers of qubits.

    This class runs a given BaseBenchmark multiple times in a sequence by looping through
    the number of qubits for initializing the BaseBenchmark. The sequential runs stop if
    an optional fail condition checked using self.should_stop is met.

    """

    def __init__(
            self,
            name: str,
            benchmark_class: type[BaseBenchmark],
            min_qubits: int = 1,
            max_qubits: int = 10,
            qubit_indices: Optional[List[int]] = None,
            fixed_parameters: Dict[str, Any] = None,
            save_path: Optional[str | Path | FileManager] = None,
    ):
        """Initialize a SequentialBenchmark instance.

        Args:
            name (str): The name of the benchmark.
            benchmark_class (type[BaseBenchmark]): A class inheriting from BaseBenchmark.
                This represents the benchmark to be run.
            min_qubits (int, optional): The minimum number of qubits to start from. Defaults to 1.
            max_qubits (int, optional): The maximum number of qubits to stop at, inclusively. Defaults to 10.
            qubit_indices (List[int], optional): The indices of the qubits to benchmark on.
                Each benchmark on n qubits will use qubits indexed by the first n-th elements of this list.
                Defaults to None.
            fixed_parameters (dict): A dictionary of fixed parameters to initialize benchmarks with.
            save_path (str | Path | FileManager, optional): Path to save benchmark outputs. Defaults to None.

        """
        super().__init__(name, qubits=[], save_path=save_path)
        self.qubits = None
        self.config["benchmark_class"] = name
        self.config.update(fixed_parameters)
        self.config["min_qubits"] = min_qubits
        self.config["max_qubits"] = max_qubits
        self.config["qubit_indices"] = qubit_indices

        self.benchmark_class = benchmark_class
        self.fixed_parameters = fixed_parameters
        self.run_index = 0
        self.benchmarks = []
        self.all_results = []

    def _get_current_qubits(self):
        """Get the qubit number or qubit indices for the current run in the sequence."""
        if self.config["qubit_indices"] is None:
            return self.config["min_qubits"] + self.run_index
        else:
            return self.config["qubit_indices"][:(self.config["min_qubits"] + self.run_index)]

    def _generate_circuits(self) -> List[QuantumCircuit] | Dict[str, any]:
        """Generate the circuits corresponding to the current run."""
        qubits = self._get_current_qubits()
        benchmark = self.benchmark_class(**self.fixed_parameters, qubits=qubits)
        circuits = benchmark._generate_circuits()
        return circuits

    def should_stop(self, _results):
        """Determine if the benchmark should be stopped based on given results.

        Default implementation never stops, and subclasses may override this
        method to customize the stopping condition.
        """
        return False

    def run(self, device: BaseDevice = None, num_shots: int = 1024, **kwargs):
        """Run the sequential benchmark for increasing qubits. Overrides BaseBenchmark.run.

        Args:
            device (BaseDevice, optional): Device to run benchmark on. Defaults to None.
            num_shots (int, optional): Number of measurements per circuit. Defaults to 1024.
            **kwargs (Dict[str, any]): Optional keyword arguments passed to device in _runtime_params.

        """
        for _ in range(self.config["min_qubits"], self.config["max_qubits"] + 1):
            qubits = self._get_current_qubits()
            benchmark = self.benchmark_class(**self.fixed_parameters, qubits=qubits)
            self.benchmarks.append(benchmark)

            results = benchmark(device, num_shots, **kwargs)
            self.all_results.append(results)

            if self.should_stop(results):
                print(f"The stopping condition is met on run {self.run_index} with qubits={qubits}.")
                break
            self.run_index += 1

    def get_largest_successful_qubit(self):
        """Get the largest qubit number with a successful run, printing suggestions if all/none runs pass."""
        largest_successful_qubit = None
        if self.run_index == 0:
            print("The first run mets the fail condition. "
                  "Please reduce the minimum number of qubits.")
        elif self.run_index == self.config["max_qubits"] + 1 - self.config["min_qubits"]:
            print("All runs finished successfully. "
                  "If there was a set fail condition, please increase the maximum number of qubits.")
        else:
            largest_successful_qubit = self.config["min_qubits"] + self.run_index - 1
            print(f"The largest qubit number with a successful run is {largest_successful_qubit}.")

        return largest_successful_qubit
