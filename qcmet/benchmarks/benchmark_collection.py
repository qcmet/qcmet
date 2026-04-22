"""benchmark_collection.py.

This module provides a wrapper class for collecting different benchmark
instances into one instance, for easier execution of the benchmarks in one go.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.result import marginal_distribution

from qcmet.benchmarks import BaseBenchmark
from qcmet.core import FileManager
from qcmet.devices import BaseDevice
from qcmet.utils.circuit_fusion import fuse_circuit_groups


class BenchmarkCollection(BaseBenchmark):
    """A collection of BaseBenchmark instances.

    This class defines a wrapper for collecting different benchmarks into one.
    This class inherits from BaseBenchmark, providing the same API as other benchmarks.

    """

    def __init__(
        self,
        benchmarks: List[BaseBenchmark] | Dict[str, BaseBenchmark],
        fuse_circuits: bool = False,
        fuse_mode: str = "strict",
        save_path: Optional[str | Path | FileManager] = None,
    ):
        """Initialize a BenchmarkCollection instance.

        Args:
            benchmarks (List[BaseBenchmark] | Dict[str, BaseBenchmark]): Either:
                - List[BaseBenchmark]: List of BaseBenchmark instances.
                - Dict[str, BaseBenchmark]: Dictionary of {identifier: BaseBenchmark}.
                The label of each benchmark will be created in the format of
                "Benchmark{index}_{benchmark.name}" if no identifiers provided or
                as "identifier" if the benchmarks are passed in as a dictionary.
            fuse_circuits (bool): Whether to fuse the circuits to run benchmarks in parallel. Defaults to False.
            fuse_mode ("strict" or "min" or "pad"):
                If fuse_circuits is set to True, how to handle benchmarks with different numbers of circuits:
                - "strict": all benchmarks must have the same number of circuits
                - "min": only fuse up to the smallest number of circuits
                - "pad": fuse all circuits; smaller benchmarks simply contribute nothing to later fused circuits
            save_path (str | Path | FileManager, optional): Path to save benchmark outputs for all benchmarks.
                Overwrites the save paths of each benchmark if provided. Defaults to None.

        """
        super().__init__("BenchmarkCollection", qubits=[], save_path=save_path)
        self.qubits = None
        if isinstance(benchmarks, list):
            self._benchmark_labels = [
                f"Benchmark{i}_{benchmark.name}"
                for i, benchmark in enumerate(benchmarks)
            ]
            self._benchmarks = benchmarks
        else:
            self._benchmark_labels = list(benchmarks.keys())
            self._benchmarks = list(benchmarks.values())

        if save_path is not None:
            for i, benchmark in enumerate(benchmarks):
                path = self.file_manager.run_path /  f"{self._benchmark_labels[i]}"
                benchmark.set_save_path(path)
                benchmark.file_manager = FileManager(benchmark.name, Path(benchmark.save_path))

        self._num_circs_per_benchmark = []
        self._runtime_params = None

        # Original circuit lists from each benchmark
        self._original_circuits: List[List[QuantumCircuit]] | None = None

        # Circuit fusion
        self._fuse_circuits = fuse_circuits
        self._fuse_mode = fuse_mode
        self._fused = False
        self._fused_circuits = None
        self._clbits = None

    @property
    def num_qubits(self) -> Dict:
        """Return the number of qubits for each benchmark in the collection. Overrides BaseBenchmark.num_qubits."""
        num_qubits = {}
        for i, benchmark in enumerate(self._benchmarks):
            num_qubits[self._benchmark_labels[i]] = benchmark.num_qubits
        return num_qubits

    def _reset_cached_state(self) -> None:
        """Reset cached grouped/fused circuit state."""
        self._num_circs_per_benchmark = []
        self._original_circuits = None
        self._fused = False
        self._fused_circuits = None
        self._clbits = None

    def _flatten_groups(self, circuit_groups: List[List[QuantumCircuit]]) -> List[QuantumCircuit]:
        """Flatten grouped circuits into a single list."""
        flat = []
        for group in circuit_groups:
            flat.extend(group)
        return flat

    def _set_collection_circuits(self, circuits: List[QuantumCircuit]) -> None:
        """Replace only the collection's own circuits.

        This does not modify the child benchmarks' stored circuits.
        """
        self.experiment_data = list(circuits)

    def _generate_circuits(self):
        """Call each benchmark in the collection to generate circuits, and collect them together.

        Returns:
            List[QuantumCircuit]: Flattened list of all child benchmark circuits.

        """
        self._reset_cached_state()

        benchmark_circuit_groups = []

        for benchmark in self._benchmarks:
            benchmark.generate_circuits()
            circuits = list(benchmark.circuits)
            benchmark_circuit_groups.append(circuits)
            self._num_circs_per_benchmark.append(len(circuits))

        self._original_circuits = benchmark_circuit_groups
        circuits = self._flatten_groups(benchmark_circuit_groups)

        if self._fuse_circuits:
            circuits = self.fuse_circuits(self._fuse_mode)

        return circuits

    def fuse_circuits(
        self,
        fuse_mode: str = "strict",
        circuit_groups: List[List[QuantumCircuit]] | None = None,
    ) -> List[QuantumCircuit]:
        """Fuse circuits across benchmarks for parallel execution.

        By default this uses each benchmark's stored circuits directly.
        `circuit_groups` is only a one-off override for advanced use cases,
        e.g., when the circuits need another transpilation pass before fusion.

        Args:
            fuse_mode: Passed to fuse_circuit_groups. One of "strict", "min", or "pad".
            circuit_groups: Optional grouped circuits to fuse instead of the benchmarks'
                own stored circuits. If provided, its group sizes must match the original
                benchmark circuit counts.

        Returns:
            List[QuantumCircuit]: The fused circuits.

        """
        if self._original_circuits is None:
            raise RuntimeError("Benchmark circuit groups are not available. Call generate_circuits() first.")

        if circuit_groups is None:
            circuit_groups = self._original_circuits
        else:
            circuit_groups = [list(group) for group in circuit_groups]
            if len(circuit_groups) != len(self._benchmarks):
                raise ValueError("circuit_groups must have the same length as the number of benchmarks.")
            for i, (orig_group, new_group) in enumerate(zip(self._original_circuits, circuit_groups, strict=True)):
                if len(orig_group) != len(new_group):
                    raise ValueError(f"circuit_groups[{i}] has length {len(new_group)}, "
                                     f"but benchmark {i} has {len(orig_group)} circuits.")

        fused_circuits, clbits = fuse_circuit_groups(circuit_groups, fuse_mode)

        self._fused = True
        self._fused_circuits = fused_circuits
        self._clbits = clbits

        # The collection itself now runs the fused circuits
        self._set_collection_circuits(fused_circuits)

        return fused_circuits

    def run(
        self,
        device: BaseDevice = None,
        num_shots: Optional[int | List[int]] = 1024,
        **kwargs,
    ):
        """Run all benchmarks in the collection. Overrides BaseBenchmark.run.

        If `num_shots` is an int:
            - if unfused, run all child circuits in one go
            - if fused, run the fused circuits in one go and later split counts

        If `num_shots` is a List[int], each benchmark is run separately with its own
        shot count, and fusion is not supported. In this case, self._experiment_data and
        self._runtime_params will be left as None, since each benchmark will have their own
        _experiment_data and _runtime_params generated by their respective run.

        Args:
            device (BaseDevice, optional): Device to run benchmark on. Defaults to None.
            num_shots (int | List[int], optional): Either
                - int: Number of shots, applied to all benchmarks in the collection.
                - List[int]: A list of number of shots for each benchmark in the collection.
                Defaults to 1024.
            **kwargs (Dict[str, any]): Optional keyword arguments passed to device in _runtime_params.

        """
        if isinstance(num_shots, int):
            super().run(device, num_shots, **kwargs)
        else:
            if self._fused:
                raise ValueError(
                    "Fusion is not supported when num_shots is provided as a list. "
                    "Use a single integer number of shots for fused runs."
                )

            if len(num_shots) != len(self._benchmarks):
                raise ValueError(
                    "The provided list of shots does not have the same size as the number of benchmarks."
                )

            self._runtime_params = None

            for i, benchmark in enumerate(self._benchmarks):
                benchmark.run(device, num_shots[i], **kwargs)

    def _distribute_unfused_counts(self) -> None:
        """Distribute flat collection measurements back into each child benchmark."""
        prev_num_circs = 0
        for i, benchmark in enumerate(self._benchmarks):
            num_circs = self._num_circs_per_benchmark[i]
            circuit_measurements = (
                self.experiment_data[prev_num_circs : prev_num_circs + num_circs]
                .reset_index(drop=True)
                .copy()
            )["circuit_measurements"].to_list()

            benchmark.load_circuit_measurements(circuit_measurements)
            benchmark._runtime_params = self._runtime_params
            prev_num_circs += num_circs

    def _distribute_fused_counts(self) -> None:
        """Split fused collection measurements back into each child benchmark."""
        fused_counts = self.experiment_data["circuit_measurements"].to_list()

        fused_counts_list = [fused_counts] if isinstance(fused_counts, dict) else list(fused_counts)

        per_group_counts = [[] for _ in self._clbits]

        for counts in fused_counts_list:
            # Convert counts into qiskit's little-endian format
            counts = {bitstring[::-1]: value for bitstring, value in counts.items()}
            for group_index, clbits in enumerate(self._clbits):
                sub_counts = dict(marginal_distribution(counts, indices=list(clbits)))
                # Convert back to big-endian
                sub_counts = {bitstring[::-1]: value for bitstring, value in sub_counts.items()}
                per_group_counts[group_index].append(sub_counts)

        for benchmark, counts in zip(self._benchmarks, per_group_counts, strict=True):
            benchmark.load_circuit_measurements(counts)
            benchmark._runtime_params = self._runtime_params

    def _analyze(self):
        """Distribute measurements to each benchmark, call analyze, and collect results.

        Returns:
            dict: A dictionary containing the results of each benchmark. Each key-value pair corresponds to
                  the name of a benchmark in the collection prepended with an index label, and its respective
                  result which is another dictionary.

        """
        all_results = {}

        if self._runtime_params is not None:
            # If during run, shots is an int, then all circuits were ran in one go
            # So we need to distribute the circuit measurements from this class to each benchmark
            if self._fused:
                self._distribute_fused_counts()
            else:
                self._distribute_unfused_counts()

        for i, benchmark in enumerate(self._benchmarks):
            result = benchmark.analyze()
            all_results[self._benchmark_labels[i]] = result

        return all_results

    def save(self):
        """Call each benchmark in the collection to save the results."""
        for benchmark in self._benchmarks:
            benchmark.save()

    def has_plotting(self):
        """Check if _plot function is implemented for at least one of the benchmarks in the collection.

        Overrides BaseBenchmark.has_plotting.

        Returns:
            bool: True if at least one of the benchmarks in the collection have a plot function.

        """
        return any(benchmark.has_plotting() for benchmark in self._benchmarks)

    def plot(self, axes: List[matplotlib.axes._axes.Axes] = None):
        """Plot benchmark results for all benchmarks in the collection that has a plot function.

        Args:
            axes (List[matplotlib.axes._axes.Axes], optional): Plot will use axes if provided.
                The length of this list must be equal to the number of benchmarks in the collection
                with plotting implemented. Defaults to None.

        """
        benchmarks_with_plotting = [
            benchmark for benchmark in self._benchmarks if benchmark.has_plotting()
        ]
        benchmark_labels = [
            self._benchmark_labels[self._benchmarks.index(b)]
            for b in benchmarks_with_plotting
        ]
        if axes is not None and isinstance(axes, list):
            if len(axes) != len(benchmarks_with_plotting):
                raise ValueError(
                    "The length of axes list must be equal to the number of "
                    "benchmarks with plotting implemented."
                )
        else:
            figsize = matplotlib.rcParams["figure.figsize"]
            _, axes = plt.subplots(
                ncols=len(benchmarks_with_plotting),
                figsize=(figsize[0] * len(benchmarks_with_plotting), figsize[1]),
            )
            # Flatten it to avoid error cause by only 1 ax returned by subplots
            axes = np.array(axes).flatten()

        for i, benchmark in enumerate(benchmarks_with_plotting):
            benchmark.plot(axes[i])
            axes[i].set_title(benchmark_labels[i])
